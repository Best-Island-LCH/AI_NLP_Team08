"""
Cross-Encoder for AI Quality Evaluation

Context(질문)와 Response(응답)를 분리하여 인코딩하고
Cross-Attention으로 상호작용을 모델링합니다.

no_hallucination 기준 개선에 효과적입니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from typing import Dict, Optional


class CrossAttentionLayer(nn.Module):
    """
    Cross-Attention 레이어
    
    Response가 Context를 참조하여 더 풍부한 표현을 생성합니다.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        
        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
        
    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: Response 표현 [batch, seq_len, hidden]
            key_value: Context 표현 [batch, seq_len, hidden]
            key_padding_mask: Context 패딩 마스크 [batch, seq_len]
            
        Returns:
            enhanced_query: 향상된 Response 표현
        """
        # Cross-Attention
        attended, _ = self.attention(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=key_padding_mask
        )
        
        # Residual + LayerNorm
        query = self.layer_norm1(query + attended)
        
        # Feed-forward + Residual + LayerNorm
        ffn_out = self.ffn(query)
        query = self.layer_norm2(query + ffn_out)
        
        return query


class CrossEncoderModel(nn.Module):
    """
    Cross-Encoder 모델
    
    Context(질문)와 Response(응답)를 각각 인코딩한 후
    Cross-Attention으로 상호작용을 모델링합니다.
    """
    
    def __init__(
        self,
        model_name: str = 'klue/roberta-base',
        num_labels: int = 9,
        num_cross_layers: int = 2,
        dropout_prob: float = 0.1,
        share_encoder: bool = True
    ):
        """
        Args:
            model_name: HuggingFace 모델 ID
            num_labels: 출력 라벨 수
            num_cross_layers: Cross-Attention 레이어 수
            dropout_prob: 드롭아웃 확률
            share_encoder: Context와 Response 인코더 공유 여부
        """
        super().__init__()
        
        self.num_labels = num_labels
        self.share_encoder = share_encoder
        
        # Context Encoder
        self.context_encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.context_encoder.config.hidden_size
        
        # Response Encoder (공유 또는 별도)
        if share_encoder:
            self.response_encoder = self.context_encoder
        else:
            self.response_encoder = AutoModel.from_pretrained(model_name)
        
        # Cross-Attention Layers
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(
                hidden_size=self.hidden_size,
                num_heads=8,
                dropout=dropout_prob
            )
            for _ in range(num_cross_layers)
        ])
        
        # Pooling + Classification
        self.pooler = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout_prob)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(self.hidden_size // 2, num_labels)
        )
        
    def forward(
        self,
        context_input_ids: torch.Tensor,
        context_attention_mask: torch.Tensor,
        response_input_ids: torch.Tensor,
        response_attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        순전파
        
        Args:
            context_input_ids: Context 입력 [batch, context_len]
            context_attention_mask: Context 마스크 [batch, context_len]
            response_input_ids: Response 입력 [batch, response_len]
            response_attention_mask: Response 마스크 [batch, response_len]
            labels: 라벨 [batch, num_labels] (선택)
            
        Returns:
            logits: 예측 로짓
            loss: 손실 (labels 제공 시)
        """
        # Context 인코딩
        context_outputs = self.context_encoder(
            input_ids=context_input_ids,
            attention_mask=context_attention_mask
        )
        context_hidden = context_outputs.last_hidden_state  # [batch, ctx_len, hidden]
        context_cls = context_hidden[:, 0, :]  # [batch, hidden]
        
        # Response 인코딩
        response_outputs = self.response_encoder(
            input_ids=response_input_ids,
            attention_mask=response_attention_mask
        )
        response_hidden = response_outputs.last_hidden_state  # [batch, resp_len, hidden]
        response_cls = response_hidden[:, 0, :]  # [batch, hidden]
        
        # Cross-Attention (Response가 Context를 참조)
        # 패딩 마스크: 1이면 무시, 0이면 attend
        context_key_padding_mask = (context_attention_mask == 0)
        
        enhanced_response = response_hidden
        for cross_layer in self.cross_attention_layers:
            enhanced_response = cross_layer(
                query=enhanced_response,
                key_value=context_hidden,
                key_padding_mask=context_key_padding_mask
            )
        
        # Cross-attended CLS
        cross_cls = enhanced_response[:, 0, :]
        
        # 다양한 표현 결합
        # - context_cls: Context의 [CLS]
        # - response_cls: Response의 원래 [CLS]
        # - cross_cls: Cross-Attention 후 [CLS]
        combined = torch.cat([context_cls, response_cls, cross_cls], dim=-1)
        
        # Pooling + Classification
        pooled = self.pooler(combined)
        logits = self.classifier(pooled)
        
        # 손실 계산
        loss = None
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        
        return {'logits': logits, 'loss': loss}


class HallucinationFocusedCrossEncoder(CrossEncoderModel):
    """
    환각(Hallucination) 탐지에 특화된 Cross-Encoder
    
    Context에 없는 정보가 Response에 있는지 확인하는 데 집중합니다.
    """
    
    def __init__(
        self,
        model_name: str = 'klue/roberta-base',
        num_labels: int = 9,
        hallucination_idx: int = 5,  # no_hallucination 인덱스
        **kwargs
    ):
        super().__init__(model_name, num_labels, **kwargs)
        
        self.hallucination_idx = hallucination_idx
        
        # 환각 전용 분류 헤드
        self.hallucination_head = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, 1)
        )
        
        # Attention 기반 일치도 계산
        self.match_attention = nn.Linear(self.hidden_size, self.hidden_size)
    
    def compute_matching_score(
        self,
        context_hidden: torch.Tensor,
        response_hidden: torch.Tensor,
        context_mask: torch.Tensor,
        response_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Context와 Response 간 토큰 레벨 일치도 계산
        
        낮은 일치도 = 환각 가능성 높음
        """
        # Attention scores
        query = self.match_attention(response_hidden)  # [batch, resp_len, hidden]
        
        # [batch, resp_len, ctx_len]
        scores = torch.bmm(query, context_hidden.transpose(1, 2))
        scores = scores / (self.hidden_size ** 0.5)
        
        # Context 패딩 마스킹
        context_mask_expanded = context_mask.unsqueeze(1)
        scores = scores.masked_fill(context_mask_expanded == 0, -1e9)
        
        # 최대 일치도 (각 response 토큰의 최고 일치 context 토큰)
        max_scores = scores.max(dim=-1)[0]  # [batch, resp_len]
        
        # Response 패딩 마스킹 후 평균
        max_scores = max_scores.masked_fill(response_mask == 0, 0)
        match_score = max_scores.sum(dim=-1) / response_mask.sum(dim=-1).float()
        
        return match_score  # [batch]
    
    def forward(
        self,
        context_input_ids: torch.Tensor,
        context_attention_mask: torch.Tensor,
        response_input_ids: torch.Tensor,
        response_attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """순전파 (환각 탐지 강화)"""
        # 기본 Cross-Encoder 출력
        outputs = super().forward(
            context_input_ids,
            context_attention_mask,
            response_input_ids,
            response_attention_mask,
            labels=None  # 손실은 아래서 별도 계산
        )
        
        logits = outputs['logits']
        
        # 손실 계산 (환각에 추가 가중치)
        loss = None
        if labels is not None:
            # 기본 BCE Loss
            base_loss = F.binary_cross_entropy_with_logits(logits, labels.float())
            
            # 환각 기준에 추가 가중치
            hall_logits = logits[:, self.hallucination_idx]
            hall_labels = labels[:, self.hallucination_idx].float()
            hall_loss = F.binary_cross_entropy_with_logits(hall_logits, hall_labels)
            
            # 결합 (환각에 50% 추가 가중치)
            loss = base_loss + 0.5 * hall_loss
        
        outputs['loss'] = loss
        return outputs


def get_cross_encoder(
    model_name: str = 'klue/roberta-base',
    num_cross_layers: int = 2,
    use_hallucination_focus: bool = False,
    **kwargs
) -> nn.Module:
    """
    Cross-Encoder 모델 생성 헬퍼
    
    Args:
        model_name: HuggingFace 모델 ID
        num_cross_layers: Cross-Attention 레이어 수
        use_hallucination_focus: 환각 특화 모델 사용 여부
        **kwargs: 추가 인자
    
    Returns:
        CrossEncoderModel 또는 HallucinationFocusedCrossEncoder
    """
    if use_hallucination_focus:
        return HallucinationFocusedCrossEncoder(
            model_name=model_name,
            num_cross_layers=num_cross_layers,
            **kwargs
        )
    else:
        return CrossEncoderModel(
            model_name=model_name,
            num_cross_layers=num_cross_layers,
            **kwargs
        )
