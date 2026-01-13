#!/usr/bin/env python
"""
wandb Sweep 실행 스크립트

하이퍼파라미터 튜닝을 위한 Bayesian optimization을 실행합니다.

사용법:
    # Sweep 생성 및 실행
    python scripts/sweep.py --create --count 30
    
    # 기존 Sweep에 Agent 추가
    python scripts/sweep.py --sweep_id <SWEEP_ID> --count 10
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import wandb


def parse_args():
    parser = argparse.ArgumentParser(description='wandb Sweep 실행')
    
    parser.add_argument('--create', action='store_true',
                        help='새 Sweep 생성')
    parser.add_argument('--sweep_id', type=str, default=None,
                        help='기존 Sweep ID')
    parser.add_argument('--count', type=int, default=30,
                        help='실행할 run 수')
    parser.add_argument('--config', type=str, default='config/sweep_config.yaml',
                        help='Sweep 설정 파일')
    parser.add_argument('--entity', type=str, default='dhj9842-hanyang-university',
                        help='wandb entity')
    parser.add_argument('--project', type=str, default='mutsa-01',
                        help='wandb project')
    
    return parser.parse_args()


def load_sweep_config(config_path: str) -> dict:
    """Sweep 설정 로드"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def train_sweep():
    """
    Sweep Agent가 호출하는 학습 함수
    
    wandb.config에서 하이퍼파라미터를 읽어 학습 실행
    """
    import torch
    import numpy as np
    import random
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    
    from src.data.preprocessing import load_data, preprocess_data, CRITERIA
    from src.data.dataset import QualityEvalDataset
    from src.data.tokenizer_utils import get_tokenizer, get_sep_token
    from src.training.trainer import create_training_args, train_model
    from src.evaluation.metrics import MetricsComputer
    
    # wandb run 초기화
    run = wandb.init()
    config = wandb.config
    
    # 시드 설정
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Config: {dict(config)}")
    
    # 데이터 로드
    data_dir = Path(__file__).parent.parent.parent / 'data'
    train_df, val_df = load_data(
        str(data_dir / 'train' / 'training_all_aggregated.csv'),
        str(data_dir / 'val' / 'validation_all_aggregated.csv')
    )
    
    # 데이터 샘플링 (Sweep은 빠른 실험을 위해 샘플링)
    sample_size = min(50000, len(train_df))
    train_df = train_df.sample(n=sample_size, random_state=seed)
    val_df = val_df.sample(n=min(10000, len(val_df)), random_state=seed)
    
    # 모델 이름
    model_name = config.get('model_name', 'klue/roberta-base')
    
    # 토크나이저
    tokenizer = get_tokenizer(model_name)
    sep_token = get_sep_token(model_name)
    
    # 전처리
    loss_type = config.get('loss_type', 'bce')
    include_soft = loss_type == 'soft_bce'
    
    train_df = preprocess_data(train_df, CRITERIA, sep_token, include_soft_labels=include_soft)
    val_df = preprocess_data(val_df, CRITERIA, sep_token, include_soft_labels=include_soft)
    
    # 데이터셋
    max_length = config.get('max_length', 128)
    
    train_dataset = QualityEvalDataset(
        train_df, tokenizer, max_length, CRITERIA, 
        use_soft_labels=(loss_type == 'soft_bce')
    )
    val_dataset = QualityEvalDataset(
        val_df, tokenizer, max_length, CRITERIA,
        use_soft_labels=(loss_type == 'soft_bce')
    )
    
    # 모델
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=9,
        problem_type="multi_label_classification"
    )
    model.to(device)
    
    # 학습 설정
    output_dir = Path(__file__).parent.parent / 'outputs' / 'sweep' / run.id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = create_training_args(
        output_dir=str(output_dir),
        num_epochs=config.get('num_epochs', 3),
        batch_size=config.get('batch_size', 32),
        learning_rate=config.get('learning_rate', 2e-5),
        warmup_ratio=config.get('warmup_ratio', 0.1),
        weight_decay=config.get('weight_decay', 0.01),
        fp16=True,
        seed=seed,
        report_to='wandb'
    )
    
    # 메트릭
    metrics_computer = MetricsComputer(criteria=CRITERIA)
    
    # 손실 함수 설정
    loss_kwargs = {}
    if loss_type == 'label_smoothing':
        loss_kwargs['smoothing'] = config.get('label_smoothing_alpha', 0.1)
    
    # 학습
    trainer, _ = train_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
        training_args=training_args,
        compute_metrics=metrics_computer,
        loss_type=loss_type,
        loss_kwargs=loss_kwargs,
        use_soft_labels=(loss_type == 'soft_bce'),
        early_stopping_patience=2
    )
    
    # 최종 평가
    eval_results = trainer.evaluate()
    
    # 주요 메트릭 로깅
    wandb.log({
        'final/macro_f1': eval_results.get('eval_macro_f1', 0),
        'final/exact_match': eval_results.get('eval_exact_match', 0),
        'final/loss': eval_results.get('eval_loss', 0),
    })
    
    print(f"Final macro_f1: {eval_results.get('eval_macro_f1', 0):.4f}")


def main():
    args = parse_args()
    
    config_path = Path(__file__).parent.parent / args.config
    
    if args.create:
        # 새 Sweep 생성
        sweep_config = load_sweep_config(str(config_path))
        
        # program 경로 수정
        sweep_config['program'] = str(Path(__file__).resolve())
        
        sweep_id = wandb.sweep(
            sweep=sweep_config,
            entity=args.entity,
            project=args.project
        )
        
        print(f"✅ Sweep 생성됨: {sweep_id}")
        print(f"   Entity: {args.entity}")
        print(f"   Project: {args.project}")
        print(f"\n다음 명령어로 Agent를 실행하세요:")
        print(f"   python scripts/sweep.py --sweep_id {sweep_id} --count {args.count}")
        
        # Agent 바로 실행
        print(f"\n또는 지금 바로 실행:")
        wandb.agent(sweep_id, function=train_sweep, count=args.count,
                   entity=args.entity, project=args.project)
    
    elif args.sweep_id:
        # 기존 Sweep에 Agent 추가
        print(f"Sweep {args.sweep_id}에 Agent 추가 ({args.count} runs)")
        
        wandb.agent(args.sweep_id, function=train_sweep, count=args.count,
                   entity=args.entity, project=args.project)
    
    else:
        print("--create 또는 --sweep_id를 지정하세요.")
        print("예: python scripts/sweep.py --create --count 30")


if __name__ == '__main__':
    main()
