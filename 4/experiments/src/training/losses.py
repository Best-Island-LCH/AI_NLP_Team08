"""
손실 함수 모듈

Soft BCE Loss, Label Smoothing BCE Loss 등을 정의합니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SoftBCELoss(nn.Module):
    """
    Soft Binary Cross-Entropy Loss
    
    평가자 투표 비율을 반영한 soft label을 사용합니다.
    
    예시:
        - hard label: [1, 0] -> "확실히 1, 확실히 0"
        - soft label: [0.67, 0.33] -> "아마 1, 아마 0"
    """
    
    def __init__(self, reduction: str = 'mean', eps: float = 1e-8):
        """
        Args:
            reduction: 'mean', 'sum', 'none' 중 하나
            eps: 수치 안정성을 위한 작은 값
        """
        super().__init__()
        self.reduction = reduction
        self.eps = eps
    
    def forward(
        self,
        logits: torch.Tensor,
        soft_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: 모델 출력 (batch, num_labels)
            soft_labels: soft label (batch, num_labels), 0~1 범위
            
        Returns:
            손실 값
        """
        # Sigmoid 적용
        probs = torch.sigmoid(logits)
        
        # Soft BCE Loss
        # L = -y * log(p) - (1-y) * log(1-p)
        loss = -soft_labels * torch.log(probs + self.eps) \
               -(1 - soft_labels) * torch.log(1 - probs + self.eps)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class LabelSmoothingBCELoss(nn.Module):
    """
    Label Smoothing Binary Cross-Entropy Loss
    
    Hard label을 부드럽게 만들어 과적합을 방지합니다.
    
    예시 (alpha=0.1):
        - 원래: y=1 -> 0.95, y=0 -> 0.05
    """
    
    def __init__(
        self,
        smoothing: float = 0.1,
        reduction: str = 'mean'
    ):
        """
        Args:
            smoothing: smoothing 정도 (0~1)
            reduction: 'mean', 'sum', 'none' 중 하나
        """
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: 모델 출력 (batch, num_labels)
            labels: hard label (batch, num_labels), 0 또는 1
            
        Returns:
            손실 값
        """
        # Label smoothing 적용
        # y=1 -> 1 - smoothing/2 = 0.95 (smoothing=0.1)
        # y=0 -> smoothing/2 = 0.05
        smoothed_labels = labels * (1 - self.smoothing) + self.smoothing / 2
        
        # BCE Loss with smoothed labels
        loss = F.binary_cross_entropy_with_logits(
            logits,
            smoothed_labels,
            reduction=self.reduction
        )
        
        return loss


class UncertaintyAwareLoss(nn.Module):
    """
    Uncertainty-Aware Loss
    
    불확실한 샘플(2:1 투표)의 손실 가중치를 낮춥니다.
    """
    
    def __init__(
        self,
        base_weight: float = 0.5,
        reduction: str = 'mean'
    ):
        """
        Args:
            base_weight: 불확실한 샘플의 최소 가중치
            reduction: 'mean', 'sum', 'none' 중 하나
        """
        super().__init__()
        self.base_weight = base_weight
        self.reduction = reduction
    
    def forward(
        self,
        logits: torch.Tensor,
        hard_labels: torch.Tensor,
        soft_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: 모델 출력 (batch, num_labels)
            hard_labels: hard label (batch, num_labels)
            soft_labels: soft label (batch, num_labels)
            
        Returns:
            손실 값
        """
        # BCE Loss 계산
        bce_loss = F.binary_cross_entropy_with_logits(
            logits,
            hard_labels,
            reduction='none'
        )
        
        # 확실성 계산: soft_label이 0.5에서 멀수록 확실
        # certainty = 1 - 2 * |soft_label - 0.5|
        # soft_label=1.0 -> certainty=1.0
        # soft_label=0.5 -> certainty=0.0
        certainty = 1 - 2 * torch.abs(soft_labels - 0.5)
        
        # 가중치 계산
        # 확실한 샘플: weight=1.0
        # 불확실한 샘플: weight=base_weight
        weights = self.base_weight + (1 - self.base_weight) * certainty
        
        # 가중치 적용
        weighted_loss = bce_loss * weights
        
        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss


class FocalLoss(nn.Module):
    """
    Focal Loss
    
    어려운 샘플에 더 집중합니다.
    클래스 불균형 문제에 효과적입니다.
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ):
        """
        Args:
            gamma: focusing parameter (높을수록 어려운 샘플에 집중)
            alpha: 클래스별 가중치
            reduction: 'mean', 'sum', 'none' 중 하나
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: 모델 출력 (batch, num_labels)
            labels: label (batch, num_labels)
            
        Returns:
            손실 값
        """
        probs = torch.sigmoid(logits)
        
        # pt: p if y=1, 1-p if y=0
        pt = labels * probs + (1 - labels) * (1 - probs)
        
        # Focal weight: (1-pt)^gamma
        focal_weight = (1 - pt) ** self.gamma
        
        # BCE Loss
        bce = F.binary_cross_entropy_with_logits(
            logits,
            labels,
            reduction='none'
        )
        
        # Focal Loss
        loss = focal_weight * bce
        
        # Alpha weighting (클래스별 가중치)
        if self.alpha is not None:
            alpha_t = labels * self.alpha + (1 - labels) * (1 - self.alpha)
            loss = alpha_t * loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (ASL)
    
    Positive/Negative 샘플에 다른 감마 값을 적용합니다.
    극심한 클래스 불균형에 효과적입니다.
    
    논문: Asymmetric Loss For Multi-Label Classification (Ben-Baruch et al., 2020)
    
    EDA 기반 권장 사용:
        - specificity (11.8:1 불균형) → gamma_neg=4, gamma_pos=0
        - interestingness (9.8:1 불균형) → gamma_neg=4, gamma_pos=0
    """
    
    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 0.0,
        clip: float = 0.05,
        reduction: str = 'mean',
        eps: float = 1e-8
    ):
        """
        Args:
            gamma_neg: Negative 샘플에 대한 focusing parameter
            gamma_pos: Positive 샘플에 대한 focusing parameter
            clip: Negative 확률 클리핑 (probability shifting)
            reduction: 'mean', 'sum', 'none' 중 하나
            eps: 수치 안정성을 위한 작은 값
        """
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.reduction = reduction
        self.eps = eps
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: 모델 출력 (batch, num_labels)
            labels: label (batch, num_labels)
            
        Returns:
            손실 값
        """
        probs = torch.sigmoid(logits)
        probs_pos = probs
        probs_neg = 1 - probs
        
        # Probability shifting (negative probability clipping)
        probs_neg = (probs_neg + self.clip).clamp(max=1)
        
        # Asymmetric focusing
        # Positive: (1-p)^gamma_pos * log(p)
        # Negative: p^gamma_neg * log(1-p)
        loss_pos = labels * ((1 - probs_pos) ** self.gamma_pos) * torch.log(probs_pos + self.eps)
        loss_neg = (1 - labels) * (probs_neg ** self.gamma_neg) * torch.log(probs_neg + self.eps)
        
        loss = -loss_pos - loss_neg
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss
    
    클래스별 샘플 수의 역수를 가중치로 사용합니다.
    
    논문: Class-Balanced Loss Based on Effective Number of Samples (Cui et al., 2019)
    
    EDA 기반 가중치 계산:
        - Positive가 많은 기준: Negative에 높은 가중치
        - effective_num = (1 - beta^n) / (1 - beta), beta = 0.9999
    """
    
    def __init__(
        self,
        samples_per_class: Optional[torch.Tensor] = None,
        beta: float = 0.9999,
        reduction: str = 'mean'
    ):
        """
        Args:
            samples_per_class: 클래스별 샘플 수 (num_labels, 2) - [negative_count, positive_count]
            beta: effective number 계산용 beta
            reduction: 'mean', 'sum', 'none' 중 하나
        """
        super().__init__()
        self.beta = beta
        self.reduction = reduction
        
        if samples_per_class is not None:
            self.register_buffer('weights', self._compute_weights(samples_per_class))
        else:
            self.weights = None
    
    def _compute_weights(self, samples_per_class: torch.Tensor) -> torch.Tensor:
        """Effective number 기반 가중치 계산"""
        effective_num = 1.0 - torch.pow(self.beta, samples_per_class)
        weights = (1.0 - self.beta) / effective_num
        weights = weights / weights.sum(dim=1, keepdim=True) * 2  # 정규화
        return weights
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: 모델 출력 (batch, num_labels)
            labels: label (batch, num_labels)
            
        Returns:
            손실 값
        """
        # BCE Loss
        bce = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        
        if self.weights is not None:
            # 가중치 적용: positive면 weights[:, 1], negative면 weights[:, 0]
            weight_pos = self.weights[:, 1].unsqueeze(0)  # (1, num_labels)
            weight_neg = self.weights[:, 0].unsqueeze(0)
            sample_weights = labels * weight_pos + (1 - labels) * weight_neg
            bce = bce * sample_weights
        
        if self.reduction == 'mean':
            return bce.mean()
        elif self.reduction == 'sum':
            return bce.sum()
        else:
            return bce


class CriterionWeightedLoss(nn.Module):
    """
    Criterion-Weighted Loss
    
    기준별로 다른 가중치를 적용합니다.
    
    EDA 기반 권장 가중치:
        - no_hallucination: 2.0 (가장 어려운 판단, 맥락 의존적)
        - consistency: 1.2 (맥락 의존적)
        - specificity, interestingness: 0.8 (불균형 심함, 단일 응답 판단 가능)
    """
    
    # 기준 순서
    CRITERIA = [
        'linguistic_acceptability', 'consistency', 'interestingness',
        'unbias', 'harmlessness', 'no_hallucination',
        'understandability', 'sensibleness', 'specificity'
    ]
    
    # EDA 기반 기본 가중치
    DEFAULT_WEIGHTS = {
        'linguistic_acceptability': 1.0,
        'consistency': 1.2,
        'interestingness': 0.8,
        'unbias': 1.0,
        'harmlessness': 1.0,
        'no_hallucination': 2.0,  # 가장 어려운 기준
        'understandability': 0.8,
        'sensibleness': 1.0,
        'specificity': 0.8,
    }
    
    def __init__(
        self,
        criterion_weights: Optional[dict] = None,
        base_loss: str = 'bce',
        reduction: str = 'mean',
        **loss_kwargs
    ):
        """
        Args:
            criterion_weights: 기준별 가중치 딕셔너리
            base_loss: 기본 손실 함수 ('bce', 'soft_bce', 'focal')
            reduction: 'mean', 'sum', 'none' 중 하나
            **loss_kwargs: 기본 손실 함수 추가 인자
        """
        super().__init__()
        self.reduction = reduction
        
        # 가중치 설정
        weights_dict = criterion_weights or self.DEFAULT_WEIGHTS
        weights = torch.tensor([weights_dict.get(c, 1.0) for c in self.CRITERIA])
        self.register_buffer('weights', weights)
        
        # 기본 손실 함수
        self.base_loss = base_loss
        self.loss_kwargs = loss_kwargs
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        soft_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            logits: 모델 출력 (batch, num_labels)
            labels: label (batch, num_labels)
            soft_labels: soft label (optional, soft_bce용)
            
        Returns:
            손실 값
        """
        # 기본 손실 계산 (reduction='none')
        if self.base_loss == 'soft_bce' and soft_labels is not None:
            probs = torch.sigmoid(logits)
            eps = self.loss_kwargs.get('eps', 1e-8)
            loss = -soft_labels * torch.log(probs + eps) - (1 - soft_labels) * torch.log(1 - probs + eps)
        elif self.base_loss == 'focal':
            gamma = self.loss_kwargs.get('gamma', 2.0)
            probs = torch.sigmoid(logits)
            pt = labels * probs + (1 - labels) * (1 - probs)
            focal_weight = (1 - pt) ** gamma
            bce = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
            loss = focal_weight * bce
        else:
            loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        
        # 기준별 가중치 적용
        weighted_loss = loss * self.weights.unsqueeze(0)
        
        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss


class MultiTaskUncertaintyLoss(nn.Module):
    """
    Multi-Task Uncertainty Loss
    
    태스크별 불확실성을 학습하여 자동으로 가중치를 조절합니다.
    
    논문: Multi-Task Learning Using Uncertainty to Weigh Losses (Kendall et al., 2018)
    
    수식: L_total = Σ (1/2σ_i² × L_i + log(σ_i))
        - σ_i가 크면: 가중치 낮음 (불확실한 태스크)
        - σ_i가 작으면: 가중치 높음 (확실한 태스크)
    """
    
    def __init__(
        self,
        num_tasks: int = 9,
        reduction: str = 'mean'
    ):
        """
        Args:
            num_tasks: 태스크 수 (기준 수)
            reduction: 'mean', 'sum', 'none' 중 하나
        """
        super().__init__()
        self.num_tasks = num_tasks
        self.reduction = reduction
        
        # log(σ²)를 학습 파라미터로 설정 (수치 안정성)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: 모델 출력 (batch, num_labels)
            labels: label (batch, num_labels)
            
        Returns:
            손실 값
        """
        # 기준별 BCE Loss (reduction='none')
        bce = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        
        # 불확실성 기반 가중치 적용
        # precision = 1/σ² = exp(-log(σ²))
        precision = torch.exp(-self.log_vars)
        
        # L = 1/2σ² × L + log(σ) = 1/2 × precision × L + 1/2 × log_var
        weighted_loss = 0.5 * precision.unsqueeze(0) * bce + 0.5 * self.log_vars.unsqueeze(0)
        
        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss
    
    def get_task_weights(self) -> torch.Tensor:
        """학습된 태스크 가중치 반환 (정규화된 precision)"""
        with torch.no_grad():
            precision = torch.exp(-self.log_vars)
            weights = precision / precision.sum()
            return weights


def get_loss_function(
    loss_type: str,
    **kwargs
) -> nn.Module:
    """
    손실 함수 팩토리
    
    Args:
        loss_type: 손실 함수 종류
            - 'bce': 기본 BCE
            - 'soft_bce': Soft BCE (평가자 투표 비율 반영)
            - 'label_smoothing': Label Smoothing BCE
            - 'uncertainty': Uncertainty-Aware Loss
            - 'focal': Focal Loss (클래스 불균형)
            - 'asl': Asymmetric Loss (극심한 불균형)
            - 'class_balanced': Class-Balanced Loss
            - 'criterion_weighted': 기준별 가중치 Loss
            - 'multitask_uncertainty': Multi-Task Uncertainty Loss
        **kwargs: 손실 함수별 추가 인자
        
    Returns:
        손실 함수 모듈
    """
    if loss_type == 'bce':
        return nn.BCEWithLogitsLoss(reduction=kwargs.get('reduction', 'mean'))
    
    elif loss_type == 'soft_bce':
        return SoftBCELoss(
            reduction=kwargs.get('reduction', 'mean'),
            eps=kwargs.get('eps', 1e-8)
        )
    
    elif loss_type == 'label_smoothing':
        return LabelSmoothingBCELoss(
            smoothing=kwargs.get('smoothing', 0.1),
            reduction=kwargs.get('reduction', 'mean')
        )
    
    elif loss_type == 'uncertainty':
        return UncertaintyAwareLoss(
            base_weight=kwargs.get('base_weight', 0.5),
            reduction=kwargs.get('reduction', 'mean')
        )
    
    elif loss_type == 'focal':
        return FocalLoss(
            gamma=kwargs.get('gamma', 2.0),
            alpha=kwargs.get('alpha', None),
            reduction=kwargs.get('reduction', 'mean')
        )
    
    elif loss_type == 'asl':
        return AsymmetricLoss(
            gamma_neg=kwargs.get('gamma_neg', 4.0),
            gamma_pos=kwargs.get('gamma_pos', 0.0),
            clip=kwargs.get('clip', 0.05),
            reduction=kwargs.get('reduction', 'mean')
        )
    
    elif loss_type == 'class_balanced':
        return ClassBalancedLoss(
            samples_per_class=kwargs.get('samples_per_class', None),
            beta=kwargs.get('beta', 0.9999),
            reduction=kwargs.get('reduction', 'mean')
        )
    
    elif loss_type == 'criterion_weighted':
        return CriterionWeightedLoss(
            criterion_weights=kwargs.get('criterion_weights', None),
            base_loss=kwargs.get('base_loss', 'bce'),
            reduction=kwargs.get('reduction', 'mean'),
            **{k: v for k, v in kwargs.items() if k not in ['criterion_weights', 'base_loss', 'reduction']}
        )
    
    elif loss_type == 'multitask_uncertainty':
        return MultiTaskUncertaintyLoss(
            num_tasks=kwargs.get('num_tasks', 9),
            reduction=kwargs.get('reduction', 'mean')
        )
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
