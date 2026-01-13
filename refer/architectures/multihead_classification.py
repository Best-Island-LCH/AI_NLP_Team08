"""
Multi-Head Classification for AI Quality Evaluation

기준별 독립 분류 헤드 + 그룹 상호작용 모델
- Criterion-specific Heads
- Group Interaction Layer
- Focal Loss
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class MultiHeadClassificationModel(nn.Module):
    """
    기준별 독립 분류 헤드 + 그룹 상호작용 모델
    """
    
    # 기준 그룹 정의
    CRITERION_GROUPS = {
        'language': ['linguistic_acceptability', 'understandability'],
        'content': ['consistency', 'sensibleness', 'no_hallucination'],
        'ethics': ['unbias', 'harmlessness'],
        'quality': ['interestingness', 'specificity']
    }
    
    CRITERIA = [
        'linguistic_acceptability', 'consistency', 'interestingness',
        'unbias', 'harmlessness', 'no_hallucination',
        'understandability', 'sensibleness', 'specificity'
    ]
    
    def __init__(
        self,
        model_name: str = 'klue/roberta-base',
        use_group_interaction: bool = True,
        dropout_prob: float = 0.1
    ):
        """
        Args:
            model_name: HuggingFace 모델 ID
            use_group_interaction: 그룹 상호작용 레이어 사용 여부
            dropout_prob: 드롭아웃 확률
        """
        super().__init__()
        
        self.use_group_interaction = use_group_interaction
        
        # BERT Encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        
        # 기준별 독립 헤드
        self.criterion_heads = nn.ModuleDict()
        for criterion in self.CRITERIA:
            self.criterion_heads[criterion] = CriterionHead(
                input_size=self.hidden_size,
                hidden_size=128,
                dropout_prob=dropout_prob
            )
        
        # 그룹별 공유 레이어
        if use_group_interaction:
            self.group_layers = nn.ModuleDict()
            for group_name, criteria in self.CRITERION_GROUPS.items():
                self.group_layers[group_name] = GroupInteractionLayer(
                    hidden_size=self.hidden_size,
                    num_criteria=len(criteria),
                    dropout_prob=dropout_prob
                )
        
        # 최종 조정 레이어 (모든 기준 간 상호작용)
        self.final_interaction = nn.Sequential(
            nn.Linear(len(self.CRITERIA), 64),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(64, len(self.CRITERIA))
        )
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        순전파
        
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
        
        Returns:
            logits: [batch, num_criteria]
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
            for group_name, criteria in self.CRITERION_GROUPS.items():
                group_output = self.group_layers[group_name](cls_output)
                for i, criterion in enumerate(criteria):
                    group_enhanced[criterion] = group_output[:, i, :]
        else:
            group_enhanced = {c: cls_output for c in self.CRITERIA}
        
        # 기준별 예측
        criterion_logits = []
        for criterion in self.CRITERIA:
            head_input = group_enhanced[criterion]
            logit = self.criterion_heads[criterion](head_input)
            criterion_logits.append(logit)
        
        # [batch, num_criteria]
        logits = torch.cat(criterion_logits, dim=-1)
        
        # 최종 상호작용 (residual)
        adjustment = self.final_interaction(logits)
        final_logits = logits + 0.1 * adjustment
        
        return final_logits


class CriterionHead(nn.Module):
    """
    개별 기준 분류 헤드
    """
    
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
    """
    그룹 내 기준 상호작용 레이어
    """
    
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
    Focal Loss: 어려운 샘플(잘 맞추지 못하는)에 더 큰 가중치
    
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


# ============================================================
# 사용 예시
# ============================================================

def example_usage():
    """Multi-Head Classification 사용 예시"""
    
    print("Multi-Head Classification Model 예시")
    print("=" * 50)
    
    model = MultiHeadClassificationModel(
        model_name='klue/roberta-base',
        use_group_interaction=True
    )
    
    print(f"모델 파라미터: {sum(p.numel() for p in model.parameters()):,}")
    
    # 기준별 헤드 파라미터
    print("\n기준별 헤드:")
    for criterion, head in model.criterion_heads.items():
        params = sum(p.numel() for p in head.parameters())
        print(f"  {criterion}: {params:,} params")
    
    # 그룹별 레이어
    if model.use_group_interaction:
        print("\n그룹별 상호작용 레이어:")
        for group_name, layer in model.group_layers.items():
            params = sum(p.numel() for p in layer.parameters())
            criteria = model.CRITERION_GROUPS[group_name]
            print(f"  {group_name} ({len(criteria)} criteria): {params:,} params")
    
    # 더미 데이터로 테스트
    batch_size = 4
    seq_len = 128
    
    input_ids = torch.randint(0, 30000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    targets = torch.randint(0, 2, (batch_size, 9)).float()
    
    # Forward
    logits = model(input_ids, attention_mask)
    print(f"\n출력 shape: {logits.shape}")
    
    # Focal Loss 테스트
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    loss = criterion(logits, targets)
    print(f"Focal Loss: {loss.item():.4f}")
    
    # 일반 BCE와 비교
    bce = nn.BCEWithLogitsLoss()
    bce_loss = bce(logits, targets)
    print(f"BCE Loss: {bce_loss.item():.4f}")


if __name__ == "__main__":
    example_usage()
