"""
Multi-Head Classification for AI Quality Evaluation

기준별 독립 분류 헤드 + 그룹 상호작용 모델
- Criterion-specific Heads
- Group Interaction Layer
- Focal Loss Support
"""

import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Dict, List, Optional


# 평가 기준 정의
CRITERIA = [
    'linguistic_acceptability', 'consistency', 'interestingness',
    'unbias', 'harmlessness', 'no_hallucination',
    'understandability', 'sensibleness', 'specificity'
]

# 기준 그룹 정의
CRITERION_GROUPS = {
    'language': ['linguistic_acceptability', 'understandability'],
    'content': ['consistency', 'sensibleness', 'no_hallucination'],
    'ethics': ['unbias', 'harmlessness'],
    'quality': ['interestingness', 'specificity']
}


class MultiHeadClassificationModel(nn.Module):
    """
    기준별 독립 분류 헤드 + 그룹 상호작용 모델
    """
    
    def __init__(
        self,
        model_name: str = 'klue/roberta-base',
        num_labels: int = 9,
        use_group_interaction: bool = True,
        dropout_prob: float = 0.1,
        head_hidden_size: int = 128
    ):
        """
        Args:
            model_name: HuggingFace 모델 ID
            num_labels: 출력 라벨 수
            use_group_interaction: 그룹 상호작용 레이어 사용 여부
            dropout_prob: 드롭아웃 확률
            head_hidden_size: 개별 헤드의 은닉층 크기
        """
        super().__init__()
        
        self.num_labels = num_labels
        self.use_group_interaction = use_group_interaction
        self.criteria = CRITERIA[:num_labels]
        
        # BERT Encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        
        # 기준별 독립 헤드
        self.criterion_heads = nn.ModuleDict()
        for criterion in self.criteria:
            self.criterion_heads[criterion] = CriterionHead(
                input_size=self.hidden_size,
                hidden_size=head_hidden_size,
                dropout_prob=dropout_prob
            )
        
        # 그룹별 공유 레이어
        if use_group_interaction:
            self.group_layers = nn.ModuleDict()
            for group_name, group_criteria in CRITERION_GROUPS.items():
                # 현재 사용하는 기준만 포함
                active_criteria = [c for c in group_criteria if c in self.criteria]
                if active_criteria:
                    self.group_layers[group_name] = GroupInteractionLayer(
                        hidden_size=self.hidden_size,
                        num_criteria=len(active_criteria),
                        dropout_prob=dropout_prob
                    )
            
            # 그룹-기준 매핑 저장
            self._build_group_mapping()
        
        # 최종 조정 레이어 (모든 기준 간 상호작용)
        self.final_interaction = nn.Sequential(
            nn.Linear(num_labels, 64),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(64, num_labels)
        )
    
    def _build_group_mapping(self):
        """그룹-기준 매핑 구축"""
        self.criterion_to_group = {}
        self.group_criterion_indices = {}
        
        for group_name, group_criteria in CRITERION_GROUPS.items():
            active_criteria = [c for c in group_criteria if c in self.criteria]
            for idx, criterion in enumerate(active_criteria):
                self.criterion_to_group[criterion] = (group_name, idx)
            self.group_criterion_indices[group_name] = active_criteria
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ):
        """
        순전파
        
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            labels: [batch, num_labels] (선택, 손실 계산용)
        
        Returns:
            logits: [batch, num_labels]
            loss: 손실값 (labels 제공 시)
        """
        # BERT 인코딩
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch, hidden]
        
        # 그룹 상호작용 적용
        if self.use_group_interaction:
            group_enhanced = {}
            for group_name, layer in self.group_layers.items():
                group_output = layer(cls_output)  # [batch, num_group_criteria, hidden]
                group_criteria = self.group_criterion_indices[group_name]
                for i, criterion in enumerate(group_criteria):
                    group_enhanced[criterion] = group_output[:, i, :]
        else:
            group_enhanced = {c: cls_output for c in self.criteria}
        
        # 기준별 예측
        criterion_logits = []
        for criterion in self.criteria:
            if criterion in group_enhanced:
                head_input = group_enhanced[criterion]
            else:
                head_input = cls_output
            logit = self.criterion_heads[criterion](head_input)
            criterion_logits.append(logit)
        
        # [batch, num_labels]
        logits = torch.cat(criterion_logits, dim=-1)
        
        # 최종 상호작용 (residual)
        adjustment = self.final_interaction(logits)
        final_logits = logits + 0.1 * adjustment
        
        # 손실 계산 (labels 제공 시)
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(final_logits, labels.float())
        
        return {'logits': final_logits, 'loss': loss}


class CriterionHead(nn.Module):
    """개별 기준 분류 헤드"""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        dropout_prob: float = 0.1
    ):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, input_size]
        Returns:
            logit: [batch, 1]
        """
        return self.classifier(x)


class GroupInteractionLayer(nn.Module):
    """그룹 내 기준 상호작용 레이어"""
    
    def __init__(
        self,
        hidden_size: int,
        num_criteria: int,
        dropout_prob: float = 0.1
    ):
        super().__init__()
        
        self.num_criteria = num_criteria
        
        # 기준별 프로젝션
        self.criterion_projections = nn.Linear(
            hidden_size, hidden_size * num_criteria
        )
        
        # 그룹 내 어텐션
        self.group_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            dropout=dropout_prob,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, cls_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cls_output: [batch, hidden]
        
        Returns:
            enhanced: [batch, num_criteria, hidden]
        """
        batch_size = cls_output.size(0)
        
        # 기준별 표현 생성
        projected = self.criterion_projections(cls_output)
        projected = projected.view(batch_size, self.num_criteria, -1)
        
        # 그룹 내 어텐션으로 상호작용
        attended, _ = self.group_attention(
            projected, projected, projected
        )
        
        # Residual + LayerNorm
        enhanced = self.layer_norm(projected + attended)
        
        return enhanced


class FocalLoss(nn.Module):
    """
    Focal Loss: 어려운 샘플에 더 큰 가중치
    
    L = -α(1-p)^γ * log(p)  (positive)
    L = -(1-α)p^γ * log(1-p)  (negative)
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Args:
            alpha: 클래스 불균형 조절 (0~1)
            gamma: 어려운 샘플 강조 정도 (>0)
            reduction: 손실 축소 방법
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: [batch, num_labels]
            targets: [batch, num_labels]
        
        Returns:
            loss: Focal loss
        """
        probs = torch.sigmoid(logits)
        
        # 클램핑
        probs = torch.clamp(probs, 1e-7, 1 - 1e-7)
        
        # Focal weight
        pos_weight = self.alpha * (1 - probs) ** self.gamma
        neg_weight = (1 - self.alpha) * probs ** self.gamma
        
        # BCE
        pos_loss = -pos_weight * targets * torch.log(probs)
        neg_loss = -neg_weight * (1 - targets) * torch.log(1 - probs)
        
        loss = pos_loss + neg_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def get_multihead_model(
    model_name: str = 'klue/roberta-base',
    use_group_interaction: bool = True,
    **kwargs
) -> MultiHeadClassificationModel:
    """
    Multi-Head Classification 모델 생성 헬퍼
    
    Args:
        model_name: HuggingFace 모델 ID
        use_group_interaction: 그룹 상호작용 사용 여부
        **kwargs: 추가 인자
    
    Returns:
        MultiHeadClassificationModel 인스턴스
    """
    return MultiHeadClassificationModel(
        model_name=model_name,
        use_group_interaction=use_group_interaction,
        **kwargs
    )
