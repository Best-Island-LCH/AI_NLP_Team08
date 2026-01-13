"""
Hierarchical Attention for AI Quality Evaluation

계층적 어텐션 모델
- Turn-level Encoding
- Turn-level Attention
- Response Gate
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import List, Tuple


class HierarchicalAttentionModel(nn.Module):
    """
    계층적 어텐션 모델
    
    구조:
      1. Turn Encoder: 각 턴을 독립적으로 인코딩
      2. Turn Attention: 턴 간 중요도 계산
      3. Classifier: 최종 분류
    """
    
    def __init__(
        self, 
        model_name: str = 'klue/roberta-base', 
        num_labels: int = 9,
        max_turns: int = 8,
        attention_heads: int = 4,
        dropout_prob: float = 0.1
    ):
        """
        Args:
            model_name: HuggingFace 모델 ID
            num_labels: 출력 라벨 수
            max_turns: 최대 턴 수
            attention_heads: 어텐션 헤드 수
            dropout_prob: 드롭아웃 확률
        """
        super().__init__()
        
        self.num_labels = num_labels
        self.max_turns = max_turns
        
        # Turn Encoder (공유 BERT)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        
        # Turn-level Attention
        self.turn_attention = TurnLevelAttention(
            hidden_size=self.hidden_size,
            num_heads=attention_heads,
            dropout_prob=dropout_prob
        )
        
        # 평가 대상 응답 강조를 위한 게이트
        self.response_gate = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.Sigmoid()
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(self.hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, num_labels)
        )
    
    def encode_turns(
        self, 
        turn_input_ids: List[torch.Tensor], 
        turn_attention_masks: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        각 턴을 독립적으로 인코딩
        
        Args:
            turn_input_ids: 턴별 input_ids 리스트
            turn_attention_masks: 턴별 attention_mask 리스트
        
        Returns:
            turn_representations: [batch_size, num_turns, hidden_size]
        """
        turn_reprs = []
        
        for input_ids, attention_mask in zip(turn_input_ids, turn_attention_masks):
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            cls_output = outputs.last_hidden_state[:, 0, :]  # [batch, hidden]
            turn_reprs.append(cls_output)
        
        # [batch, num_turns, hidden]
        turn_representations = torch.stack(turn_reprs, dim=1)
        
        return turn_representations
    
    def forward(
        self, 
        turn_input_ids: List[torch.Tensor],
        turn_attention_masks: List[torch.Tensor],
        turn_mask: torch.Tensor = None
    ):
        """
        순전파
        
        Args:
            turn_input_ids: 턴별 input_ids 리스트 (각각 [batch, seq_len])
            turn_attention_masks: 턴별 attention_mask 리스트
            turn_mask: 유효한 턴 마스크 [batch, num_turns]
        
        Returns:
            logits: [batch_size, num_labels]
            attention_weights: [batch_size, num_turns] (디버깅/해석용)
        """
        # 턴 인코딩
        turn_reprs = self.encode_turns(turn_input_ids, turn_attention_masks)
        
        # 턴 어텐션
        context_repr, attention_weights = self.turn_attention(
            turn_reprs, turn_mask
        )
        
        # 마지막 턴(평가 대상)과 컨텍스트 결합
        response_repr = turn_reprs[:, -1, :]  # 마지막 턴
        
        # 게이트로 응답 강조
        gate_input = torch.cat([context_repr, response_repr], dim=-1)
        gate = self.response_gate(gate_input)
        final_repr = gate * response_repr + (1 - gate) * context_repr
        
        # 분류
        logits = self.classifier(final_repr)
        
        return logits, attention_weights


class TurnLevelAttention(nn.Module):
    """
    턴 레벨 어텐션 모듈
    
    각 턴의 중요도를 계산하여 가중 평균
    """
    
    def __init__(
        self, 
        hidden_size: int, 
        num_heads: int = 4, 
        dropout_prob: float = 0.1
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # Multi-head Self-Attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout_prob,
            batch_first=True
        )
        
        # Query for attention pooling
        self.attention_query = nn.Parameter(
            torch.randn(1, 1, hidden_size)
        )
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(
        self, 
        turn_reprs: torch.Tensor, 
        turn_mask: torch.Tensor = None
    ):
        """
        Args:
            turn_reprs: [batch, num_turns, hidden]
            turn_mask: [batch, num_turns] 유효한 턴 마스크 (True = 유효)
        
        Returns:
            context_repr: [batch, hidden]
            attention_weights: [batch, num_turns]
        """
        batch_size = turn_reprs.size(0)
        
        # Self-attention으로 턴 간 상호작용
        attn_output, _ = self.self_attention(
            turn_reprs, turn_reprs, turn_reprs,
            key_padding_mask=~turn_mask if turn_mask is not None else None
        )
        attn_output = self.layer_norm(attn_output + turn_reprs)
        
        # Attention pooling
        query = self.attention_query.expand(batch_size, -1, -1)
        
        # 어텐션 스코어 계산
        scores = torch.bmm(query, attn_output.transpose(1, 2))  # [batch, 1, num_turns]
        scores = scores.squeeze(1)  # [batch, num_turns]
        
        # 마스킹
        if turn_mask is not None:
            scores = scores.masked_fill(~turn_mask, float('-inf'))
        
        # 소프트맥스
        attention_weights = F.softmax(scores, dim=-1)
        
        # 가중 평균
        context_repr = torch.bmm(
            attention_weights.unsqueeze(1), attn_output
        ).squeeze(1)  # [batch, hidden]
        
        return context_repr, attention_weights


class HierarchicalDataProcessor:
    """
    계층적 모델을 위한 데이터 처리기
    
    대화를 턴 단위로 분리하여 처리
    """
    
    def __init__(
        self, 
        tokenizer, 
        max_turn_length: int = 128, 
        max_turns: int = 8
    ):
        """
        Args:
            tokenizer: HuggingFace 토크나이저
            max_turn_length: 각 턴의 최대 토큰 수
            max_turns: 최대 턴 수
        """
        self.tokenizer = tokenizer
        self.max_turn_length = max_turn_length
        self.max_turns = max_turns
    
    def process_conversation(
        self, 
        utterances: List[dict]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        """
        대화를 턴 단위로 처리
        
        Args:
            utterances: 발화 리스트 
                [{'speaker': 'human'|'bot', 'text': '...'}, ...]
        
        Returns:
            turn_input_ids: 턴별 input_ids 리스트
            turn_attention_masks: 턴별 attention_mask 리스트
            turn_mask: 유효 턴 마스크
        """
        turn_input_ids = []
        turn_attention_masks = []
        
        # 최근 max_turns개 턴만 사용
        recent_utterances = utterances[-self.max_turns:]
        
        for utt in recent_utterances:
            speaker = "[Human]" if utt['speaker'] == 'human' else "[Bot]"
            text = f"{speaker} {utt['text']}"
            
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_turn_length,
                return_tensors='pt'
            )
            
            turn_input_ids.append(encoding['input_ids'].squeeze(0))
            turn_attention_masks.append(encoding['attention_mask'].squeeze(0))
        
        # 패딩 (턴 수가 max_turns보다 적으면)
        num_turns = len(turn_input_ids)
        while len(turn_input_ids) < self.max_turns:
            turn_input_ids.insert(0, torch.zeros(self.max_turn_length, dtype=torch.long))
            turn_attention_masks.insert(0, torch.zeros(self.max_turn_length, dtype=torch.long))
        
        # 유효 턴 마스크
        turn_mask = torch.zeros(self.max_turns, dtype=torch.bool)
        turn_mask[-num_turns:] = True
        
        return turn_input_ids, turn_attention_masks, turn_mask


# ============================================================
# 사용 예시
# ============================================================

def example_usage():
    """계층적 모델 사용 예시"""
    
    print("Hierarchical Attention Model 예시")
    print("=" * 50)
    
    tokenizer = AutoTokenizer.from_pretrained('klue/roberta-base')
    
    model = HierarchicalAttentionModel(
        model_name='klue/roberta-base',
        num_labels=9,
        max_turns=8
    )
    
    print(f"모델 파라미터: {sum(p.numel() for p in model.parameters()):,}")
    
    processor = HierarchicalDataProcessor(
        tokenizer=tokenizer,
        max_turn_length=128,
        max_turns=8
    )
    
    # 예시 대화
    utterances = [
        {'speaker': 'human', 'text': '오늘 날씨 어때?'},
        {'speaker': 'bot', 'text': '오늘 서울은 맑고 기온이 20도입니다.'},
        {'speaker': 'human', 'text': '비 올 확률은?'},
        {'speaker': 'bot', 'text': '비 올 확률은 10% 미만으로 매우 낮습니다.'}  # 평가 대상
    ]
    
    # 처리
    turn_ids, turn_masks, valid_mask = processor.process_conversation(utterances)
    
    # 배치 차원 추가
    turn_ids = [ids.unsqueeze(0) for ids in turn_ids]
    turn_masks = [mask.unsqueeze(0) for mask in turn_masks]
    valid_mask = valid_mask.unsqueeze(0)
    
    # 추론
    model.eval()
    with torch.no_grad():
        logits, attn_weights = model(turn_ids, turn_masks, valid_mask)
        probs = torch.sigmoid(logits)
    
    print(f"\n예측 확률: {probs.numpy().round(3)}")
    print(f"턴 어텐션 가중치: {attn_weights.numpy().round(3)}")
    
    # 어텐션 해석
    print("\n어텐션 해석:")
    for i, w in enumerate(attn_weights[0]):
        if valid_mask[0, i]:
            idx = i - (8 - len(utterances))
            if idx >= 0:
                speaker = utterances[idx]['speaker']
                print(f"  Turn {idx+1} ({speaker}): {w:.3f}")


if __name__ == "__main__":
    example_usage()
