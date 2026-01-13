"""
Uncertainty Modeling for AI Quality Evaluation

평가자 불일치를 활용한 불확실성 모델링
- Soft Labels
- Label Smoothing
- Uncertainty-Aware Loss
"""

import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Optional


class SoftBCELoss(nn.Module):
    """
    Soft Label을 위한 BCE Loss
    
    일반 BCE: -[y*log(p) + (1-y)*log(1-p)]
    → y가 0 또는 1이 아닌 연속값 지원
    """
    
    def __init__(self, reduction: str = 'mean'):
        """
        Args:
            reduction: 손실 축소 방법 ('mean', 'sum', 'none')
        """
        super().__init__()
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: 모델 출력 (sigmoid 전) [batch, num_labels]
            soft_targets: 소프트 라벨 [0, 1] 범위의 연속값 [batch, num_labels]
        
        Returns:
            loss: BCE Loss
        """
        probs = torch.sigmoid(logits)
        
        # 수치 안정성을 위한 클램핑
        probs = torch.clamp(probs, 1e-7, 1 - 1e-7)
        
        # BCE 계산
        loss = -soft_targets * torch.log(probs) - \
               (1 - soft_targets) * torch.log(1 - probs)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class LabelSmoothingBCELoss(nn.Module):
    """
    Label Smoothing을 적용한 BCE Loss
    
    Hard label을 약간 부드럽게 만들어 과신 방지
    """
    
    def __init__(self, smoothing: float = 0.1, reduction: str = 'mean'):
        """
        Args:
            smoothing: 스무딩 정도 (0~1)
            reduction: 손실 축소 방법
        """
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: 모델 출력
            targets: Hard labels (0 또는 1)
        
        Returns:
            loss: Label-smoothed BCE Loss
        """
        # Label smoothing 적용
        # y=1 → (1-α) + α/2 = 1 - α/2
        # y=0 → 0 + α/2 = α/2
        smooth_targets = targets * (1 - self.smoothing) + self.smoothing / 2
        
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, 1e-7, 1 - 1e-7)
        
        loss = -smooth_targets * torch.log(probs) - \
               (1 - smooth_targets) * torch.log(1 - probs)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class UncertaintyAwareLoss(nn.Module):
    """
    불확실성 인식 손실 함수
    
    불확실한 샘플(평가자 의견 분분)은 손실 가중치를 낮춤
    """
    
    def __init__(self, reduction: str = 'mean'):
        """
        Args:
            reduction: 손실 축소 방법
        """
        super().__init__()
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(
        self, 
        logits: torch.Tensor, 
        hard_targets: torch.Tensor, 
        soft_targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: 모델 출력
            hard_targets: 다수결 라벨 (0 또는 1)
            soft_targets: 투표 비율 (0~1 연속값)
        
        불확실성 계산:
            soft_target이 0.5에 가까울수록 불확실
            uncertainty = 1 - |soft_target - 0.5| * 2
        
        Returns:
            loss: Uncertainty-weighted BCE Loss
        """
        # 불확실성 계산 (0.5에 가까우면 높음)
        uncertainty = 1 - torch.abs(soft_targets - 0.5) * 2
        
        # 확실성 (가중치로 사용)
        certainty = 1 - uncertainty
        
        # BCE Loss
        loss = self.bce(logits, hard_targets)
        
        # 확실성으로 가중치 적용 (확실할수록 손실 높게)
        # 불확실한 샘플도 완전히 무시하지 않음 (0.5 유지)
        weighted_loss = loss * (0.5 + 0.5 * certainty)
        
        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss


class UncertaintyAwareModel(nn.Module):
    """
    불확실성을 명시적으로 출력하는 모델
    
    출력:
      - 예측값 (mean)
      - 불확실성 (variance)
    """
    
    def __init__(self, model_name: str = 'klue/roberta-base', num_labels: int = 9):
        """
        Args:
            model_name: HuggingFace 모델 ID
            num_labels: 출력 라벨 수
        """
        super().__init__()
        
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        
        # Mean head (예측값)
        self.mean_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_labels)
        )
        
        # Variance head (log variance 출력)
        self.var_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_labels)
        )
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        순전파
        
        Returns:
            mean: 예측 평균 [batch, num_labels]
            log_var: 예측 log variance [batch, num_labels]
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        mean = self.mean_head(cls_output)
        log_var = self.var_head(cls_output)
        
        return mean, log_var
    
    def predict_with_uncertainty(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ):
        """
        예측값과 불확실성 함께 반환
        
        Returns:
            probs: 예측 확률 [batch, num_labels]
            uncertainty: 불확실성 [batch, num_labels]
        """
        mean, log_var = self.forward(input_ids, attention_mask)
        
        probs = torch.sigmoid(mean)
        uncertainty = torch.exp(log_var)  # variance
        
        return probs, uncertainty


class GaussianNLLLoss(nn.Module):
    """
    Gaussian Negative Log-Likelihood Loss
    
    평균과 분산을 동시에 학습
    """
    
    def forward(
        self, 
        mean: torch.Tensor, 
        log_var: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        L = 0.5 * (log(var) + (target - mean)² / var)
        
        Args:
            mean: 예측 평균
            log_var: 예측 log variance
            targets: 실제 라벨
        
        Returns:
            loss: Gaussian NLL Loss
        """
        var = torch.exp(log_var)
        
        loss = 0.5 * (log_var + (targets - torch.sigmoid(mean)) ** 2 / var)
        
        return loss.mean()


class SoftLabelDataset(torch.utils.data.Dataset):
    """
    Soft Label을 지원하는 데이터셋
    """
    
    CRITERIA = [
        'linguistic_acceptability', 'consistency', 'interestingness',
        'unbias', 'harmlessness', 'no_hallucination',
        'understandability', 'sensibleness', 'specificity'
    ]
    
    def __init__(
        self, 
        samples: list, 
        tokenizer, 
        max_length: int = 512, 
        use_soft_labels: bool = True
    ):
        """
        Args:
            samples: 샘플 리스트
            tokenizer: 토크나이저
            max_length: 최대 시퀀스 길이
            use_soft_labels: Soft label 사용 여부
        """
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_soft_labels = use_soft_labels
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        
        encoding = self.tokenizer(
            sample['input_text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Hard labels (다수결)
        hard_labels = []
        soft_labels = []
        
        for c in self.CRITERIA:
            # Hard label
            if f'{c}_majority' in sample:
                hard_labels.append(float(sample[f'{c}_majority']))
            else:
                hard_labels.append(0.0)
            
            # Soft label
            if self.use_soft_labels:
                yes_count = sample.get(f'{c}_yes_count', 0)
                no_count = sample.get(f'{c}_no_count', 0)
                total = yes_count + no_count
                if total > 0:
                    soft_labels.append(yes_count / total)
                else:
                    soft_labels.append(0.5)
            else:
                soft_labels.append(hard_labels[-1])
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'hard_labels': torch.tensor(hard_labels, dtype=torch.float),
            'soft_labels': torch.tensor(soft_labels, dtype=torch.float)
        }


# ============================================================
# 사용 예시
# ============================================================

def example_usage():
    """Uncertainty Modeling 사용 예시"""
    
    print("Uncertainty Modeling 예시")
    print("=" * 50)
    
    # 더미 데이터
    batch_size = 4
    num_labels = 9
    
    logits = torch.randn(batch_size, num_labels)
    hard_labels = torch.randint(0, 2, (batch_size, num_labels)).float()
    soft_labels = torch.rand(batch_size, num_labels)  # 0~1 연속값
    
    # 1. Soft BCE Loss
    print("\n1. Soft BCE Loss")
    criterion1 = SoftBCELoss()
    loss1 = criterion1(logits, soft_labels)
    print(f"   Loss: {loss1.item():.4f}")
    
    # 2. Label Smoothing
    print("\n2. Label Smoothing BCE Loss (α=0.1)")
    criterion2 = LabelSmoothingBCELoss(smoothing=0.1)
    loss2 = criterion2(logits, hard_labels)
    print(f"   Loss: {loss2.item():.4f}")
    
    # 3. Uncertainty-Aware Loss
    print("\n3. Uncertainty-Aware Loss")
    criterion3 = UncertaintyAwareLoss()
    loss3 = criterion3(logits, hard_labels, soft_labels)
    print(f"   Loss: {loss3.item():.4f}")
    
    # 4. 비교: 일반 BCE
    print("\n4. 일반 BCE Loss (비교용)")
    criterion4 = nn.BCEWithLogitsLoss()
    loss4 = criterion4(logits, hard_labels)
    print(f"   Loss: {loss4.item():.4f}")
    
    # 5. Uncertainty-Aware Model
    print("\n5. Uncertainty-Aware Model")
    print("   (모델 로드 생략)")


def example_training():
    """실제 학습에서의 사용 예시"""
    
    # 손실 함수 선택
    loss_type = 'soft_bce'  # 'soft_bce', 'label_smoothing', 'uncertainty_aware'
    
    if loss_type == 'soft_bce':
        criterion = SoftBCELoss()
        print("Using: Soft BCE Loss")
    elif loss_type == 'label_smoothing':
        criterion = LabelSmoothingBCELoss(smoothing=0.1)
        print("Using: Label Smoothing BCE Loss")
    elif loss_type == 'uncertainty_aware':
        criterion = UncertaintyAwareLoss()
        print("Using: Uncertainty-Aware Loss")
    
    # 학습 루프 예시 (의사 코드)
    """
    for batch in train_loader:
        logits = model(batch['input_ids'], batch['attention_mask'])
        
        if loss_type == 'soft_bce':
            loss = criterion(logits, batch['soft_labels'])
        elif loss_type == 'label_smoothing':
            loss = criterion(logits, batch['hard_labels'])
        elif loss_type == 'uncertainty_aware':
            loss = criterion(logits, batch['hard_labels'], batch['soft_labels'])
        
        loss.backward()
        optimizer.step()
    """


if __name__ == "__main__":
    example_usage()
