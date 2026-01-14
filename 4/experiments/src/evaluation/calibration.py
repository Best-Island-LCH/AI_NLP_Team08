"""
Calibration 지표 모듈

ECE (Expected Calibration Error), Brier Score 등을 계산합니다.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt


def compute_ece(
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Expected Calibration Error (ECE) 계산
    
    예측 확률과 실제 정답률의 차이를 측정합니다.
    
    Args:
        predictions: 예측 확률 (samples,) 또는 (samples, num_labels)
        labels: 실제 라벨 (samples,) 또는 (samples, num_labels)
        n_bins: bin 개수
        
    Returns:
        ECE 값 (0~1, 낮을수록 좋음)
    """
    # 1D로 flatten
    predictions = predictions.flatten()
    labels = labels.flatten()
    
    # Bin 경계
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    ece = 0.0
    total_samples = len(predictions)
    
    for i in range(n_bins):
        # 현재 bin에 속하는 샘플
        in_bin = (predictions > bin_boundaries[i]) & (predictions <= bin_boundaries[i + 1])
        
        if in_bin.sum() > 0:
            # Bin 내 평균 confidence
            avg_confidence = predictions[in_bin].mean()
            
            # Bin 내 실제 accuracy
            avg_accuracy = labels[in_bin].mean()
            
            # Bin 비율
            prop_in_bin = in_bin.sum() / total_samples
            
            # ECE 누적
            ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin
    
    return float(ece)


def compute_ece_per_criterion(
    predictions: np.ndarray,
    labels: np.ndarray,
    criteria: List[str],
    n_bins: int = 10
) -> Dict[str, float]:
    """
    기준별 ECE 계산
    
    Args:
        predictions: 예측 확률 (samples, num_labels)
        labels: 실제 라벨 (samples, num_labels)
        criteria: 평가 기준 리스트
        n_bins: bin 개수
        
    Returns:
        기준별 ECE 딕셔너리
    """
    ece_per_criterion = {}
    
    for i, criterion in enumerate(criteria):
        ece = compute_ece(predictions[:, i], labels[:, i], n_bins)
        ece_per_criterion[criterion] = ece
    
    return ece_per_criterion


def compute_brier_score(
    predictions: np.ndarray,
    labels: np.ndarray
) -> float:
    """
    Brier Score 계산
    
    예측 확률과 실제 라벨의 MSE를 측정합니다.
    
    Args:
        predictions: 예측 확률 (samples,) 또는 (samples, num_labels)
        labels: 실제 라벨 (samples,) 또는 (samples, num_labels)
        
    Returns:
        Brier Score (0~1, 낮을수록 좋음)
    """
    # Flatten
    predictions = predictions.flatten()
    labels = labels.flatten()
    
    # MSE
    brier = np.mean((predictions - labels) ** 2)
    
    return float(brier)


def compute_brier_score_per_criterion(
    predictions: np.ndarray,
    labels: np.ndarray,
    criteria: List[str]
) -> Dict[str, float]:
    """
    기준별 Brier Score 계산
    
    Args:
        predictions: 예측 확률 (samples, num_labels)
        labels: 실제 라벨 (samples, num_labels)
        criteria: 평가 기준 리스트
        
    Returns:
        기준별 Brier Score 딕셔너리
    """
    brier_per_criterion = {}
    
    for i, criterion in enumerate(criteria):
        brier = compute_brier_score(predictions[:, i], labels[:, i])
        brier_per_criterion[criterion] = brier
    
    return brier_per_criterion


def compute_reliability_diagram_data(
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reliability Diagram을 위한 데이터 계산
    
    Args:
        predictions: 예측 확률
        labels: 실제 라벨
        n_bins: bin 개수
        
    Returns:
        (bin_centers, bin_accuracies, bin_counts) 튜플
    """
    predictions = predictions.flatten()
    labels = labels.flatten()
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    
    bin_accuracies = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    
    for i in range(n_bins):
        in_bin = (predictions > bin_boundaries[i]) & (predictions <= bin_boundaries[i + 1])
        
        if in_bin.sum() > 0:
            bin_accuracies[i] = labels[in_bin].mean()
            bin_counts[i] = in_bin.sum()
    
    return bin_centers, bin_accuracies, bin_counts


def plot_reliability_diagram(
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
    title: str = "Reliability Diagram",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Reliability Diagram 시각화
    
    Args:
        predictions: 예측 확률
        labels: 실제 라벨
        n_bins: bin 개수
        title: 그래프 제목
        save_path: 저장 경로 (None이면 저장하지 않음)
        
    Returns:
        matplotlib Figure
    """
    bin_centers, bin_accuracies, bin_counts = compute_reliability_diagram_data(
        predictions, labels, n_bins
    )
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    # Reliability Diagram
    ax1.bar(bin_centers, bin_accuracies, width=1/n_bins, alpha=0.7, label='Model')
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax1.set_xlabel('Mean Predicted Probability')
    ax1.set_ylabel('Fraction of Positives')
    ax1.set_title(title)
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    # ECE 표시
    ece = compute_ece(predictions, labels, n_bins)
    ax1.text(0.05, 0.95, f'ECE = {ece:.4f}', transform=ax1.transAxes, 
             verticalalignment='top', fontsize=12, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Histogram
    ax2.bar(bin_centers, bin_counts, width=1/n_bins, alpha=0.7, color='gray')
    ax2.set_xlabel('Mean Predicted Probability')
    ax2.set_ylabel('Count')
    ax2.set_xlim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def compute_calibration_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    criteria: List[str],
    n_bins: int = 10
) -> Dict[str, float]:
    """
    전체 calibration 메트릭 계산
    
    Args:
        predictions: 예측 확률 (samples, num_labels)
        labels: 실제 라벨 (samples, num_labels)
        criteria: 평가 기준 리스트
        n_bins: bin 개수
        
    Returns:
        메트릭 딕셔너리
    """
    metrics = {}
    
    # 전체 ECE 및 Brier Score
    metrics['ece'] = compute_ece(predictions, labels, n_bins)
    metrics['brier_score'] = compute_brier_score(predictions, labels)
    
    # 기준별 ECE
    ece_per_criterion = compute_ece_per_criterion(predictions, labels, criteria, n_bins)
    for criterion, ece in ece_per_criterion.items():
        metrics[f'{criterion}_ece'] = ece
    
    # 기준별 Brier Score
    brier_per_criterion = compute_brier_score_per_criterion(predictions, labels, criteria)
    for criterion, brier in brier_per_criterion.items():
        metrics[f'{criterion}_brier'] = brier
    
    return metrics
