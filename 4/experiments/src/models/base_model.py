"""
기본 분류 모델

Multi-label Classification을 위한 BERT 기반 모델을 정의합니다.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForSequenceClassification, AutoConfig
from typing import Optional, Dict, Any


class QualityClassificationModel(nn.Module):
    """
    AI 품질 평가 Multi-label Classification 모델
    
    사전학습된 BERT 계열 모델 위에 분류 헤드를 추가합니다.
    """
    
    def __init__(
        self,
        model_name: str = 'klue/roberta-base',
        num_labels: int = 9,
        dropout_prob: float = 0.1,
        use_custom_head: bool = False
    ):
        """
        Args:
            model_name: HuggingFace 모델 ID
            num_labels: 출력 라벨 수 (9개 기준)
            dropout_prob: 드롭아웃 확률
            use_custom_head: 커스텀 분류 헤드 사용 여부
        """
        super().__init__()
        
        self.model_name = model_name
        self.num_labels = num_labels
        self.use_custom_head = use_custom_head
        
        if use_custom_head:
            # 기본 모델만 로드하고 커스텀 헤드 사용
            self.encoder = AutoModel.from_pretrained(model_name)
            self.hidden_size = self.encoder.config.hidden_size
            
            # 커스텀 분류 헤드
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_prob),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.GELU(),
                nn.Dropout(dropout_prob),
                nn.Linear(self.hidden_size, num_labels)
            )
        else:
            # HuggingFace의 기본 분류 모델 사용
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                problem_type="multi_label_classification"
            )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        순전파
        
        Args:
            input_ids: 입력 토큰 ID (batch, seq_len)
            attention_mask: 어텐션 마스크 (batch, seq_len)
            labels: 라벨 (batch, num_labels), 선택적
            
        Returns:
            {'logits': ..., 'loss': ...} 딕셔너리
        """
        if self.use_custom_head:
            # 커스텀 헤드
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # [CLS] 토큰 표현
            pooled_output = outputs.last_hidden_state[:, 0, :]
            logits = self.classifier(pooled_output)
            
            result = {'logits': logits}
            
            # 손실 계산 (labels가 있으면)
            if labels is not None:
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
                result['loss'] = loss
        else:
            # HuggingFace 기본 모델
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            result = {'logits': outputs.logits}
            if labels is not None:
                result['loss'] = outputs.loss
        
        return result
    
    def get_encoder(self):
        """인코더 반환 (임베딩 추출용)"""
        if self.use_custom_head:
            return self.encoder
        else:
            return self.model.base_model


def load_model(
    model_name: str,
    num_labels: int = 9,
    device: Optional[torch.device] = None,
    **kwargs
) -> QualityClassificationModel:
    """
    모델 로드 유틸리티
    
    Args:
        model_name: HuggingFace 모델 ID
        num_labels: 라벨 수
        device: 디바이스 (None이면 자동 감지)
        **kwargs: 추가 인자
        
    Returns:
        모델 객체
    """
    model = QualityClassificationModel(
        model_name=model_name,
        num_labels=num_labels,
        **kwargs
    )
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    
    return model


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    모델 파라미터 수 계산
    
    Args:
        model: PyTorch 모델
        
    Returns:
        파라미터 수 딕셔너리
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable
    }
