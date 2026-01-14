"""
분석 모듈

Confusion Matrix, conversation_id별 분석 등을 수행합니다.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from sklearn.metrics import confusion_matrix, f1_score


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


def compute_confusion_matrices(
    predictions: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
    criteria: List[str] = CRITERIA
) -> Dict[str, np.ndarray]:
    """
    기준별 Confusion Matrix 계산
    
    Args:
        predictions: 예측 확률 (samples, num_labels)
        labels: 실제 라벨 (samples, num_labels)
        threshold: 이진화 임계값
        criteria: 평가 기준 리스트
        
    Returns:
        기준별 confusion matrix 딕셔너리
        각 matrix: [[TN, FP], [FN, TP]]
    """
    binary_preds = (predictions > threshold).astype(int)
    labels = labels.astype(int)
    
    confusion_matrices = {}
    
    for i, criterion in enumerate(criteria):
        cm = confusion_matrix(labels[:, i], binary_preds[:, i], labels=[0, 1])
        confusion_matrices[criterion] = cm
    
    return confusion_matrices


def confusion_matrix_to_dict(cm: np.ndarray) -> Dict[str, int]:
    """
    Confusion matrix를 딕셔너리로 변환
    
    Args:
        cm: confusion matrix [[TN, FP], [FN, TP]]
        
    Returns:
        {'TN': ..., 'FP': ..., 'FN': ..., 'TP': ...}
    """
    return {
        'TN': int(cm[0, 0]),
        'FP': int(cm[0, 1]),
        'FN': int(cm[1, 0]),
        'TP': int(cm[1, 1])
    }


def analyze_conversation_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    conversation_ids: List,
    threshold: float = 0.5,
    criteria: List[str] = CRITERIA
) -> Dict[str, float]:
    """
    conversation_id 기반 평가 메트릭 계산
    
    Args:
        predictions: 예측 확률 (samples, num_labels)
        labels: 실제 라벨 (samples, num_labels)
        conversation_ids: conversation_id 리스트
        threshold: 이진화 임계값
        criteria: 평가 기준 리스트
        
    Returns:
        대화별 집계 메트릭
    """
    binary_preds = (predictions > threshold).astype(int)
    labels = labels.astype(int)
    
    # conversation_id별로 그룹화
    conv_groups = defaultdict(list)
    for idx, conv_id in enumerate(conversation_ids):
        conv_groups[conv_id].append(idx)
    
    # 대화별 메트릭 계산
    conv_exact_match_list = []  # 대화 내 모든 턴이 완벽히 맞은 비율
    conv_f1_list = []  # 대화별 F1
    conv_accuracy_list = []  # 대화별 accuracy
    
    for conv_id, indices in conv_groups.items():
        conv_preds = binary_preds[indices]
        conv_labels = labels[indices]
        
        # 대화 내 완전 일치율 (모든 턴의 모든 기준이 맞아야 함)
        turn_exact = np.all(conv_preds == conv_labels, axis=1)
        conv_exact = np.all(turn_exact)
        conv_exact_match_list.append(conv_exact)
        
        # 대화별 F1 (모든 턴의 평균)
        conv_f1 = f1_score(
            conv_labels.flatten(),
            conv_preds.flatten(),
            average='micro',
            zero_division=0
        )
        conv_f1_list.append(conv_f1)
        
        # 대화별 accuracy
        conv_acc = (conv_preds == conv_labels).mean()
        conv_accuracy_list.append(conv_acc)
    
    metrics = {
        'num_conversations': len(conv_groups),
        'conv_exact_match': np.mean(conv_exact_match_list),
        'conv_avg_f1': np.mean(conv_f1_list),
        'conv_avg_accuracy': np.mean(conv_accuracy_list),
        'conv_f1_std': np.std(conv_f1_list),
    }
    
    return metrics


def analyze_errors(
    predictions: np.ndarray,
    labels: np.ndarray,
    texts: List[str],
    threshold: float = 0.5,
    criteria: List[str] = CRITERIA,
    top_k: int = 10
) -> Dict[str, List[Dict]]:
    """
    오분류 샘플 분석
    
    Args:
        predictions: 예측 확률
        labels: 실제 라벨
        texts: 입력 텍스트 리스트
        threshold: 이진화 임계값
        criteria: 평가 기준 리스트
        top_k: 각 기준별 반환할 오류 샘플 수
        
    Returns:
        기준별 오분류 샘플 딕셔너리
    """
    binary_preds = (predictions > threshold).astype(int)
    labels = labels.astype(int)
    
    error_analysis = {}
    
    for i, criterion in enumerate(criteria):
        errors = []
        
        for idx in range(len(texts)):
            pred = binary_preds[idx, i]
            label = labels[idx, i]
            prob = predictions[idx, i]
            
            if pred != label:
                errors.append({
                    'index': idx,
                    'text': texts[idx][:200] + '...' if len(texts[idx]) > 200 else texts[idx],
                    'prediction': pred,
                    'probability': float(prob),
                    'label': label,
                    'error_type': 'FP' if pred == 1 else 'FN'
                })
        
        # 확률 기준으로 정렬 (가장 확신있게 틀린 것)
        errors.sort(key=lambda x: abs(x['probability'] - 0.5), reverse=True)
        error_analysis[criterion] = errors[:top_k]
    
    return error_analysis


def compute_criterion_correlations(
    predictions: np.ndarray,
    labels: np.ndarray,
    criteria: List[str] = CRITERIA
) -> pd.DataFrame:
    """
    기준 간 상관관계 분석
    
    Args:
        predictions: 예측 확률 (samples, num_labels)
        labels: 실제 라벨 (samples, num_labels)
        criteria: 평가 기준 리스트
        
    Returns:
        상관관계 DataFrame
    """
    # 예측값 기준 상관관계
    pred_df = pd.DataFrame(predictions, columns=criteria)
    pred_corr = pred_df.corr()
    
    return pred_corr


def generate_evaluation_report(
    predictions: np.ndarray,
    labels: np.ndarray,
    conversation_ids: Optional[List] = None,
    threshold: float = 0.5,
    criteria: List[str] = CRITERIA
) -> Dict:
    """
    종합 평가 리포트 생성
    
    Args:
        predictions: 예측 확률
        labels: 실제 라벨
        conversation_ids: conversation_id 리스트 (선택)
        threshold: 이진화 임계값
        criteria: 평가 기준 리스트
        
    Returns:
        종합 리포트 딕셔너리
    """
    from .metrics import compute_metrics, compute_per_criterion_metrics
    from .calibration import compute_calibration_metrics
    from .threshold import find_optimal_thresholds
    
    report = {
        'overall_metrics': compute_metrics(predictions, labels, threshold, criteria),
        'per_criterion_metrics': compute_per_criterion_metrics(predictions, labels, threshold, criteria),
        'calibration_metrics': compute_calibration_metrics(predictions, labels, criteria),
        'confusion_matrices': {
            c: confusion_matrix_to_dict(cm) 
            for c, cm in compute_confusion_matrices(predictions, labels, threshold, criteria).items()
        },
        'optimal_thresholds': find_optimal_thresholds(predictions, labels, criteria),
    }
    
    # conversation_id가 있으면 대화별 메트릭 추가
    if conversation_ids is not None:
        report['conversation_metrics'] = analyze_conversation_metrics(
            predictions, labels, conversation_ids, threshold, criteria
        )
    
    return report
