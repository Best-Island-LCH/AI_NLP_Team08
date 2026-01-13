#!/usr/bin/env python
"""
모델 평가 스크립트

학습된 모델을 로드하여 상세 평가를 수행합니다.

사용법:
    python scripts/evaluate.py --model_path outputs/klue-roberta-base/best_model
    python scripts/evaluate.py --model_path outputs/klue-roberta-base/best_model --optimize_threshold
"""

import os
import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.preprocessing import load_data, preprocess_data, CRITERIA
from src.data.dataset import QualityEvalDataset, collate_fn
from src.data.tokenizer_utils import get_sep_token
from src.evaluation.metrics import compute_metrics, compute_per_criterion_metrics
from src.evaluation.calibration import compute_calibration_metrics, plot_reliability_diagram
from src.evaluation.threshold import find_optimal_thresholds, compare_thresholds
from src.evaluation.analysis import (
    compute_confusion_matrices, 
    analyze_conversation_metrics,
    generate_evaluation_report
)


def parse_args():
    parser = argparse.ArgumentParser(description='모델 평가')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='학습된 모델 경로')
    parser.add_argument('--data_path', type=str, default=None,
                        help='평가 데이터 경로 (기본: validation)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='결과 저장 디렉토리')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--optimize_threshold', action='store_true',
                        help='기준별 최적 threshold 탐색')
    parser.add_argument('--save_predictions', action='store_true',
                        help='예측 결과 저장')
    
    return parser.parse_args()


def predict(model, dataloader, device):
    """모델 예측 수행"""
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_conversation_ids = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].numpy()
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.cpu().numpy()
            probs = 1 / (1 + np.exp(-logits))  # Sigmoid
            
            all_predictions.append(probs)
            all_labels.append(labels)
            
            if 'conversation_id' in batch:
                all_conversation_ids.extend(batch['conversation_id'])
    
    predictions = np.vstack(all_predictions)
    labels = np.vstack(all_labels)
    
    return predictions, labels, all_conversation_ids if all_conversation_ids else None


def main():
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 모델 및 토크나이저 로드
    print(f"\n모델 로드: {args.model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model.to(device)
    
    # 모델 이름 추정 (config에서)
    model_name = model.config._name_or_path
    sep_token = get_sep_token(model_name)
    
    # 데이터 로드
    data_dir = Path(__file__).parent.parent.parent / 'data'
    
    if args.data_path:
        val_df = pd.read_csv(args.data_path, encoding='utf-8-sig')
    else:
        val_df = pd.read_csv(data_dir / 'val' / 'validation_all_aggregated.csv', encoding='utf-8-sig')
    
    print(f"평가 데이터: {len(val_df):,} samples")
    
    # 전처리
    val_df = preprocess_data(val_df, CRITERIA, sep_token, include_soft_labels=False)
    
    # 데이터셋 및 데이터로더
    val_dataset = QualityEvalDataset(
        val_df, tokenizer, args.max_length, CRITERIA, use_soft_labels=False
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # 예측
    print("\n예측 수행 중...")
    predictions, labels, conversation_ids = predict(model, val_dataloader, device)
    
    # 출력 디렉토리 설정
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.model_path) / 'evaluation'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========================================
    # 1. 기본 메트릭 (threshold=0.5)
    # ========================================
    print("\n" + "=" * 60)
    print("1. 기본 메트릭 (threshold=0.5)")
    print("=" * 60)
    
    metrics = compute_metrics(predictions, labels, threshold=0.5, criteria=CRITERIA)
    
    print(f"\n전체 메트릭:")
    print(f"  Exact Match: {metrics['exact_match']:.4f}")
    print(f"  Micro F1:    {metrics['micro_f1']:.4f}")
    print(f"  Macro F1:    {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1: {metrics['weighted_f1']:.4f}")
    
    # ========================================
    # 2. 기준별 상세 메트릭
    # ========================================
    print("\n" + "=" * 60)
    print("2. 기준별 상세 메트릭")
    print("=" * 60)
    
    per_criterion = compute_per_criterion_metrics(predictions, labels, threshold=0.5, criteria=CRITERIA)
    
    print(f"\n{'기준':<30} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 70)
    for criterion, cmetrics in per_criterion.items():
        print(f"{criterion:<30} {cmetrics['accuracy']:>10.4f} {cmetrics['precision']:>10.4f} "
              f"{cmetrics['recall']:>10.4f} {cmetrics['f1']:>10.4f}")
    
    # ========================================
    # 3. Calibration 메트릭
    # ========================================
    print("\n" + "=" * 60)
    print("3. Calibration 메트릭")
    print("=" * 60)
    
    calibration_metrics = compute_calibration_metrics(predictions, labels, CRITERIA)
    
    print(f"\n전체:")
    print(f"  ECE:         {calibration_metrics['ece']:.4f}")
    print(f"  Brier Score: {calibration_metrics['brier_score']:.4f}")
    
    # Reliability Diagram 저장
    fig = plot_reliability_diagram(
        predictions, labels, n_bins=10,
        title="Reliability Diagram",
        save_path=str(output_dir / 'reliability_diagram.png')
    )
    print(f"\nReliability Diagram 저장: {output_dir / 'reliability_diagram.png'}")
    
    # ========================================
    # 4. Threshold 최적화 (선택)
    # ========================================
    if args.optimize_threshold:
        print("\n" + "=" * 60)
        print("4. Threshold 최적화")
        print("=" * 60)
        
        comparison = compare_thresholds(predictions, labels, CRITERIA)
        
        print(f"\n{'기준':<30} {'Fixed F1':>10} {'Opt Thresh':>10} {'Opt F1':>10} {'Improve':>10}")
        print("-" * 70)
        
        total_improvement = 0
        for criterion, result in comparison.items():
            improve = result['improvement']
            total_improvement += improve
            print(f"{criterion:<30} {result['fixed_f1']:>10.4f} {result['optimal_threshold']:>10.2f} "
                  f"{result['optimal_f1']:>10.4f} {improve:>+10.4f}")
        
        print("-" * 70)
        print(f"평균 F1 개선: {total_improvement / len(CRITERIA):+.4f}")
    
    # ========================================
    # 5. Conversation 레벨 메트릭 (있는 경우)
    # ========================================
    if conversation_ids:
        print("\n" + "=" * 60)
        print("5. Conversation 레벨 메트릭")
        print("=" * 60)
        
        conv_metrics = analyze_conversation_metrics(
            predictions, labels, conversation_ids, threshold=0.5, criteria=CRITERIA
        )
        
        print(f"\n대화 수: {conv_metrics['num_conversations']:,}")
        print(f"대화별 Exact Match: {conv_metrics['conv_exact_match']:.4f}")
        print(f"대화별 평균 F1:     {conv_metrics['conv_avg_f1']:.4f} (±{conv_metrics['conv_f1_std']:.4f})")
    
    # ========================================
    # 결과 저장
    # ========================================
    print("\n" + "=" * 60)
    print("결과 저장")
    print("=" * 60)
    
    # 전체 리포트 생성
    report = generate_evaluation_report(
        predictions=predictions,
        labels=labels,
        conversation_ids=conversation_ids,
        threshold=0.5,
        criteria=CRITERIA
    )
    
    # numpy 배열 변환
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj
    
    report = convert_numpy(report)
    
    # 저장
    report_path = output_dir / 'evaluation_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"평가 리포트 저장: {report_path}")
    
    # 예측 결과 저장 (선택)
    if args.save_predictions:
        predictions_path = output_dir / 'predictions.npz'
        np.savez(
            predictions_path,
            predictions=predictions,
            labels=labels,
            criteria=CRITERIA
        )
        print(f"예측 결과 저장: {predictions_path}")
    
    print("\n✅ 평가 완료!")


if __name__ == '__main__':
    main()
