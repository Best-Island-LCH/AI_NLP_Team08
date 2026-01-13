"""
Threshold 최적화 모듈

기준별 최적 threshold를 탐색합니다.
"""

import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import f1_score, precision_score, recall_score


def find_optimal_threshold(
    predictions: np.ndarray,
    labels: np.ndarray,
    threshold_range: Tuple[float, float] = (0.3, 0.7),
    threshold_step: float = 0.05,
    metric: str = 'f1'
) -> Dict[str, float]:
    """
    단일 기준에 대한 최적 threshold 찾기
    
    Args:
        predictions: 예측 확률 (samples,)
        labels: 실제 라벨 (samples,)
        threshold_range: 탐색 범위
        threshold_step: 탐색 단위
        metric: 최적화할 메트릭 ('f1', 'precision', 'recall')
        
    Returns:
        최적 threshold 정보 딕셔너리
    """
    labels = labels.astype(int)
    
    thresholds = np.arange(threshold_range[0], threshold_range[1] + threshold_step, threshold_step)
    
    best_score = -1
    best_threshold = 0.5
    best_metrics = {}
    
    for thresh in thresholds:
        binary_preds = (predictions > thresh).astype(int)
        
        if metric == 'f1':
            score = f1_score(labels, binary_preds, zero_division=0)
        elif metric == 'precision':
            score = precision_score(labels, binary_preds, zero_division=0)
        elif metric == 'recall':
            score = recall_score(labels, binary_preds, zero_division=0)
        else:
            score = f1_score(labels, binary_preds, zero_division=0)
        
        if score > best_score:
            best_score = score
            best_threshold = thresh
            best_metrics = {
                'threshold': float(thresh),
                'f1': f1_score(labels, binary_preds, zero_division=0),
                'precision': precision_score(labels, binary_preds, zero_division=0),
                'recall': recall_score(labels, binary_preds, zero_division=0),
            }
    
    return best_metrics


def find_optimal_thresholds(
    predictions: np.ndarray,
    labels: np.ndarray,
    criteria: List[str],
    threshold_range: Tuple[float, float] = (0.3, 0.7),
    threshold_step: float = 0.05,
    metric: str = 'f1'
) -> Dict[str, Dict[str, float]]:
    """
    기준별 최적 threshold 찾기
    
    Args:
        predictions: 예측 확률 (samples, num_labels)
        labels: 실제 라벨 (samples, num_labels)
        criteria: 평가 기준 리스트
        threshold_range: 탐색 범위
        threshold_step: 탐색 단위
        metric: 최적화할 메트릭
        
    Returns:
        기준별 최적 threshold 정보 딕셔너리
    """
    optimal_thresholds = {}
    
    for i, criterion in enumerate(criteria):
        optimal_thresholds[criterion] = find_optimal_threshold(
            predictions[:, i],
            labels[:, i],
            threshold_range,
            threshold_step,
            metric
        )
    
    return optimal_thresholds


def apply_optimal_thresholds(
    predictions: np.ndarray,
    optimal_thresholds: Dict[str, Dict[str, float]],
    criteria: List[str]
) -> np.ndarray:
    """
    최적 threshold를 적용하여 이진화
    
    Args:
        predictions: 예측 확률 (samples, num_labels)
        optimal_thresholds: 기준별 최적 threshold
        criteria: 평가 기준 리스트
        
    Returns:
        이진화된 예측 (samples, num_labels)
    """
    binary_preds = np.zeros_like(predictions)
    
    for i, criterion in enumerate(criteria):
        thresh = optimal_thresholds.get(criterion, {}).get('threshold', 0.5)
        binary_preds[:, i] = (predictions[:, i] > thresh).astype(int)
    
    return binary_preds


def compare_thresholds(
    predictions: np.ndarray,
    labels: np.ndarray,
    criteria: List[str],
    fixed_threshold: float = 0.5
) -> Dict[str, Dict[str, float]]:
    """
    고정 threshold와 최적 threshold 비교
    
    Args:
        predictions: 예측 확률
        labels: 실제 라벨
        criteria: 평가 기준 리스트
        fixed_threshold: 고정 threshold
        
    Returns:
        비교 결과 딕셔너리
    """
    results = {}
    
    # 최적 threshold 찾기
    optimal = find_optimal_thresholds(predictions, labels, criteria)
    
    for i, criterion in enumerate(criteria):
        # 고정 threshold
        fixed_preds = (predictions[:, i] > fixed_threshold).astype(int)
        fixed_f1 = f1_score(labels[:, i].astype(int), fixed_preds, zero_division=0)
        
        # 최적 threshold
        opt_thresh = optimal[criterion]['threshold']
        opt_f1 = optimal[criterion]['f1']
        
        results[criterion] = {
            'fixed_threshold': fixed_threshold,
            'fixed_f1': fixed_f1,
            'optimal_threshold': opt_thresh,
            'optimal_f1': opt_f1,
            'improvement': opt_f1 - fixed_f1
        }
    
    return results
