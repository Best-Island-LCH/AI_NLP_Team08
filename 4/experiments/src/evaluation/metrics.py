"""
평가 지표 모듈

기준별 F1, Precision, Recall 등 다양한 메트릭을 계산합니다.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    classification_report
)

# 평가 기준
CRITERIA = [
    'linguistic_acceptability',
    'consistency',
    'interestingness',
    'unbias',
    'harmlessness',
    'no_hallucination',
    'understandability',
    'sensibleness',
    'specificity'
]


def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
    criteria: List[str] = CRITERIA
) -> Dict[str, float]:
    """
    전체 평가 지표 계산
    
    Args:
        predictions: 예측 확률 (samples, num_labels)
        labels: 실제 라벨 (samples, num_labels)
        threshold: 이진화 임계값
        criteria: 평가 기준 리스트
        
    Returns:
        메트릭 딕셔너리
    """
    # Sigmoid가 아직 적용되지 않은 경우 적용
    if predictions.min() < 0 or predictions.max() > 1:
        predictions = 1 / (1 + np.exp(-predictions))
    
    # 이진화
    binary_preds = (predictions > threshold).astype(int)
    labels = labels.astype(int)
    
    # Exact match (모든 라벨이 일치해야 정답)
    exact_match = np.all(binary_preds == labels, axis=1).mean()
    
    # Micro/Macro F1
    micro_f1 = f1_score(labels, binary_preds, average='micro')
    macro_f1 = f1_score(labels, binary_preds, average='macro')
    weighted_f1 = f1_score(labels, binary_preds, average='weighted')
    
    # 샘플별 F1 평균
    samples_f1 = f1_score(labels, binary_preds, average='samples')
    
    metrics = {
        'exact_match': exact_match,
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'samples_f1': samples_f1,
    }
    
    # 기준별 메트릭 추가
    per_criterion = compute_per_criterion_metrics(
        predictions, labels, threshold, criteria
    )
    
    for criterion, criterion_metrics in per_criterion.items():
        for metric_name, value in criterion_metrics.items():
            metrics[f'{criterion}_{metric_name}'] = value
    
    return metrics


def compute_per_criterion_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
    criteria: List[str] = CRITERIA
) -> Dict[str, Dict[str, float]]:
    """
    기준별 상세 평가 지표 계산
    
    Args:
        predictions: 예측 확률 (samples, num_labels)
        labels: 실제 라벨 (samples, num_labels)
        threshold: 이진화 임계값
        criteria: 평가 기준 리스트
        
    Returns:
        기준별 메트릭 딕셔너리
    """
    # Sigmoid가 아직 적용되지 않은 경우 적용
    if predictions.min() < 0 or predictions.max() > 1:
        predictions = 1 / (1 + np.exp(-predictions))
    
    binary_preds = (predictions > threshold).astype(int)
    labels = labels.astype(int)
    
    per_criterion_metrics = {}
    
    for i, criterion in enumerate(criteria):
        y_true = labels[:, i]
        y_pred = binary_preds[:, i]
        y_prob = predictions[:, i]
        
        # 기본 메트릭
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
        }
        
        # Positive 비율
        metrics['pos_rate_true'] = y_true.mean()
        metrics['pos_rate_pred'] = y_pred.mean()
        
        per_criterion_metrics[criterion] = metrics
    
    return per_criterion_metrics


def compute_metrics_with_optimal_threshold(
    predictions: np.ndarray,
    labels: np.ndarray,
    criteria: List[str] = CRITERIA,
    threshold_range: Tuple[float, float] = (0.3, 0.7),
    threshold_step: float = 0.05
) -> Dict[str, float]:
    """
    기준별 최적 threshold를 사용한 평가 지표 계산
    
    Args:
        predictions: 예측 확률
        labels: 실제 라벨
        criteria: 평가 기준 리스트
        threshold_range: threshold 탐색 범위
        threshold_step: threshold 탐색 단위
        
    Returns:
        메트릭 딕셔너리
    """
    from .threshold import find_optimal_thresholds
    
    # Sigmoid가 아직 적용되지 않은 경우 적용
    if predictions.min() < 0 or predictions.max() > 1:
        predictions = 1 / (1 + np.exp(-predictions))
    
    # 최적 threshold 찾기
    optimal_thresholds = find_optimal_thresholds(
        predictions, labels, criteria, threshold_range, threshold_step
    )
    
    # 기준별 최적 threshold로 이진화
    binary_preds = np.zeros_like(predictions)
    for i, criterion in enumerate(criteria):
        thresh = optimal_thresholds.get(criterion, {}).get('threshold', 0.5)
        binary_preds[:, i] = (predictions[:, i] > thresh).astype(int)
    
    labels = labels.astype(int)
    
    # 메트릭 계산
    exact_match = np.all(binary_preds == labels, axis=1).mean()
    micro_f1 = f1_score(labels, binary_preds, average='micro')
    macro_f1 = f1_score(labels, binary_preds, average='macro')
    weighted_f1 = f1_score(labels, binary_preds, average='weighted')
    
    metrics = {
        'exact_match_opt': exact_match,
        'micro_f1_opt': micro_f1,
        'macro_f1_opt': macro_f1,
        'weighted_f1_opt': weighted_f1,
    }
    
    # 기준별 최적 threshold 및 F1 추가
    for criterion, thresh_info in optimal_thresholds.items():
        metrics[f'{criterion}_opt_threshold'] = thresh_info['threshold']
        metrics[f'{criterion}_opt_f1'] = thresh_info['f1']
    
    return metrics


class MetricsComputer:
    """
    HuggingFace Trainer용 compute_metrics 함수 클래스
    """
    
    def __init__(
        self,
        criteria: List[str] = CRITERIA,
        threshold: float = 0.5,
        optimize_threshold: bool = False
    ):
        self.criteria = criteria
        self.threshold = threshold
        self.optimize_threshold = optimize_threshold
    
    def __call__(self, eval_pred) -> Dict[str, float]:
        """
        Args:
            eval_pred: (predictions, labels) 튜플
            
        Returns:
            메트릭 딕셔너리
        """
        predictions, labels = eval_pred
        
        # Sigmoid 적용
        predictions = 1 / (1 + np.exp(-predictions))
        
        # 기본 메트릭 계산
        metrics = compute_metrics(
            predictions, labels, self.threshold, self.criteria
        )
        
        # 최적 threshold 메트릭 추가
        if self.optimize_threshold:
            opt_metrics = compute_metrics_with_optimal_threshold(
                predictions, labels, self.criteria
            )
            metrics.update(opt_metrics)
        
        return metrics
