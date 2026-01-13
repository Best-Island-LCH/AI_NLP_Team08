"""
Multi-task Learning for AI Quality Evaluation

다중 태스크 학습을 적용한 AI 품질 평가 모델
- Shared Encoder + Task-specific Heads
- Uncertainty-based Loss Weighting
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class MultiTaskQualityModel(nn.Module):
    """
    Multi-task Learning 기반 AI 품질 평가 모델
    
    구조:
      Shared Encoder → Task-specific Heads (9개)
                    → Interaction Layer (선택적)
    """
    
    def __init__(
        self, 
        model_name: str = 'klue/roberta-base', 
        num_criteria: int = 9, 
        use_interaction: bool = True,
        dropout_prob: float = 0.1
    ):
        """
        Args:
            model_name: HuggingFace 모델 ID
            num_criteria: 평가 기준 수
            use_interaction: 기준 간 상호작용 레이어 사용 여부
            dropout_prob: 드롭아웃 확률
        """
        super().__init__()
        
        self.num_criteria = num_criteria
        self.use_interaction = use_interaction
        
        # Shared Encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        
        # Task-specific Heads (각 기준별 독립적인 분류기)
        self.task_heads = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_size, 128),
                nn.GELU(),
                nn.Dropout(dropout_prob),
                nn.Linear(128, 1)  # 각 기준별 1개 출력
            )
            for _ in range(num_criteria)
        ])
        
        # Criterion Interaction Layer (선택적)
        if use_interaction:
            self.interaction_layer = CriterionInteractionLayer(
                num_criteria=num_criteria,
                hidden_dim=64
            )
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        순전파
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        
        Returns:
            logits: [batch_size, num_criteria]
        """
        # Shared encoding
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch, hidden]
        
        # 각 기준별 예측
        task_outputs = []
        for head in self.task_heads:
            out = head(cls_output)  # [batch, 1]
            task_outputs.append(out)
        
        # [batch, num_criteria]
        logits = torch.cat(task_outputs, dim=1)
        
        # Interaction Layer 적용 (선택적)
        if self.use_interaction:
            logits = self.interaction_layer(logits)
        
        return logits


class CriterionInteractionLayer(nn.Module):
    """
    기준 간 상호작용을 모델링하는 레이어
    
    아이디어: 기준들 간의 관계를 학습하여 서로 보완
    예: consistency가 높으면 sensibleness도 높을 가능성 반영
    """
    
    def __init__(self, num_criteria: int = 9, hidden_dim: int = 64):
        """
        Args:
            num_criteria: 평가 기준 수
            hidden_dim: 숨겨진 차원
        """
        super().__init__()
        
        self.num_criteria = num_criteria
        
        # 기준 간 attention
        self.criterion_attention = nn.MultiheadAttention(
            embed_dim=1,  # 각 기준의 로짓은 1차원
            num_heads=1,
            batch_first=True
        )
        
        # 기준 임베딩 (각 기준의 특성을 학습)
        self.criterion_embeddings = nn.Parameter(
            torch.randn(num_criteria, hidden_dim)
        )
        
        # 상호작용 후 조정
        self.adjustment = nn.Sequential(
            nn.Linear(num_criteria + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_criteria)
        )
    
    def forward(self, logits: torch.Tensor):
        """
        Args:
            logits: [batch_size, num_criteria]
        
        Returns:
            adjusted_logits: [batch_size, num_criteria]
        """
        batch_size = logits.size(0)
        
        # 기준 임베딩과 로짓 결합
        criterion_emb = self.criterion_embeddings.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # [batch, num_criteria, hidden_dim]
        
        # 로짓을 기준 임베딩과 concat
        logits_expanded = logits.unsqueeze(-1)  # [batch, num_criteria, 1]
        combined = torch.cat([
            logits_expanded.expand(-1, -1, criterion_emb.size(-1)),
            criterion_emb
        ], dim=-1)  # [batch, num_criteria, hidden_dim + 1]
        
        # 평균 후 adjustment
        combined_mean = combined.mean(dim=1)  # [batch, hidden_dim + 1]
        
        # 원래 로짓과 concat 후 조정
        adjustment_input = torch.cat([logits, combined_mean], dim=-1)
        adjustment = self.adjustment(adjustment_input)
        
        # Residual connection
        adjusted_logits = logits + 0.1 * adjustment
        
        return adjusted_logits


class MultiTaskLoss(nn.Module):
    """
    Multi-task Learning 손실 함수
    
    여러 가중치 전략 지원:
    1. 균등 가중치 (equal)
    2. 고정 가중치 (fixed)
    3. 불확실성 가중치 (uncertainty) - Kendall et al., 2018
    """
    
    def __init__(
        self, 
        num_tasks: int = 9, 
        weighting: str = 'equal', 
        initial_weights: torch.Tensor = None
    ):
        """
        Args:
            num_tasks: 태스크(기준) 수
            weighting: 가중치 전략 ('equal', 'fixed', 'uncertainty')
            initial_weights: 고정 가중치 사용 시 초기 가중치
        """
        super().__init__()
        
        self.num_tasks = num_tasks
        self.weighting = weighting
        
        if weighting == 'equal':
            self.weights = nn.Parameter(
                torch.ones(num_tasks) / num_tasks,
                requires_grad=False
            )
        elif weighting == 'fixed':
            if initial_weights is None:
                initial_weights = torch.ones(num_tasks)
            self.weights = nn.Parameter(
                initial_weights / initial_weights.sum(),
                requires_grad=False
            )
        elif weighting == 'uncertainty':
            # 불확실성 기반 가중치 (Kendall et al., 2018)
            # log(σ²)를 학습하여 가중치로 사용
            self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            logits: [batch_size, num_tasks]
            targets: [batch_size, num_tasks]
        
        Returns:
            total_loss: 스칼라
            task_losses: 각 태스크별 손실 (디버깅용)
        """
        # 각 태스크별 BCE Loss
        task_losses = self.bce(logits, targets).mean(dim=0)  # [num_tasks]
        
        if self.weighting == 'uncertainty':
            # 불확실성 가중치: L_total = Σ (1/2σ² × L_i + log(σ))
            precision = torch.exp(-self.log_vars)  # 1/σ²
            total_loss = (precision * task_losses + self.log_vars).sum()
        else:
            total_loss = (self.weights * task_losses).sum()
        
        return total_loss, task_losses
    
    def get_learned_weights(self):
        """학습된 가중치 반환 (uncertainty 모드)"""
        if self.weighting == 'uncertainty':
            weights = torch.exp(-self.log_vars)
            return weights / weights.sum()
        else:
            return self.weights


# ============================================================
# 사용 예시
# ============================================================

def example_usage():
    """Multi-task Learning 사용 예시"""
    
    # 1. 모델 생성
    model = MultiTaskQualityModel(
        model_name='klue/roberta-base',
        num_criteria=9,
        use_interaction=True
    )
    
    print(f"모델 파라미터: {sum(p.numel() for p in model.parameters()):,}")
    
    # 2. 손실 함수 (불확실성 가중치)
    criterion = MultiTaskLoss(
        num_tasks=9,
        weighting='uncertainty'
    )
    
    # 3. 더미 데이터
    batch_size = 4
    seq_len = 128
    
    input_ids = torch.randint(0, 30000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    targets = torch.randint(0, 2, (batch_size, 9)).float()
    
    # 4. Forward pass
    logits = model(input_ids, attention_mask)
    loss, task_losses = criterion(logits, targets)
    
    print(f"Total Loss: {loss.item():.4f}")
    print(f"Task Losses: {task_losses.detach().numpy()}")
    
    # 5. 학습된 가중치 확인
    weights = criterion.get_learned_weights()
    
    criteria_names = [
        'linguistic', 'consistency', 'interesting',
        'unbias', 'harmless', 'no_halluc',
        'understand', 'sensible', 'specific'
    ]
    
    print("\n학습된 태스크 가중치:")
    for name, w in zip(criteria_names, weights):
        print(f"  {name}: {w:.4f}")


if __name__ == "__main__":
    example_usage()
