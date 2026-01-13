#!/usr/bin/env python
"""
Curriculum Learning 학습 스크립트

쉬운 샘플부터 시작하여 점진적으로 어려운 샘플을 포함합니다.

사용법:
    python scripts/train_curriculum.py --strategy linear
    python scripts/train_curriculum.py --strategy sqrt
    python scripts/train_curriculum.py --strategy step
"""

import os
import sys
import argparse

# GPU 설정 (torch 임포트 전에 반드시 실행)
def _set_gpu_early():
    for i, arg in enumerate(sys.argv):
        if arg == '--gpu' and i + 1 < len(sys.argv):
            os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[i + 1]
            print(f"GPU 설정: {sys.argv[i + 1]}")
            break

_set_gpu_early()

import torch
import numpy as np
from pathlib import Path
from datetime import datetime

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoModelForSequenceClassification, TrainingArguments
import wandb

from src.data.preprocessing import (
    load_data, preprocess_data, preprocess_data_with_context, 
    print_label_distribution, CRITERIA
)
from src.data.curriculum_dataset import CurriculumDataset
from src.data.tokenizer_utils import get_tokenizer, get_sep_token
from src.training.curriculum_trainer import train_with_curriculum, CurriculumTrainer
from src.evaluation.metrics import MetricsComputer
from src.utils.wandb_utils import init_wandb, finish_wandb
from src.utils.config_utils import (
    load_config,
    merge_configs,
    get_training_config,
    get_early_stopping_config,
    get_loss_config,
    get_context_config,
    set_seed,
    get_reproducibility_config,
    add_common_args,
    setup_experiment,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Curriculum Learning 학습')
    parser = add_common_args(parser)
    
    # Curriculum 설정
    parser.add_argument('--strategy', type=str, default='linear',
                        choices=['linear', 'sqrt', 'step'],
                        help='난이도 증가 전략')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 설정 초기화
    config = setup_experiment(args)
    
    # 디바이스
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # wandb 초기화
    wandb_config = config.get('wandb', {})
    wandb_enabled = wandb_config.get('enabled', True) and not args.no_wandb
    
    if wandb_enabled:
        loss_type = config.get('loss', {}).get('type', 'soft_bce')
        context_config_for_wandb = get_context_config(config)
        use_context_for_wandb = context_config_for_wandb['enabled'] and not args.no_context
        context_tag = 'with_context' if use_context_for_wandb else 'no_context'
        run_name = args.run_name or f"curriculum-{args.strategy}-{context_tag}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        init_wandb(
            config=config,
            entity=wandb_config.get('entity', 'dhj9842-hanyang-university'),
            project=wandb_config.get('project', 'mutsa-01'),
            name=run_name,
            tags=['curriculum', args.strategy, config['model']['name'].split('/')[-1], loss_type, context_tag]
        )
    
    # 데이터 로드
    data_config = config.get('data', {})
    data_dir = Path(__file__).parent.parent.parent / 'data'
    
    train_path = data_config.get('train_path', str(data_dir / 'train' / 'training_all_aggregated.csv'))
    val_path = data_config.get('val_path', str(data_dir / 'val' / 'validation_all_aggregated.csv'))
    
    if not Path(train_path).is_absolute():
        train_path = str(Path(__file__).parent.parent / train_path)
    if not Path(val_path).is_absolute():
        val_path = str(Path(__file__).parent.parent / val_path)
    
    train_df, val_df = load_data(train_path, val_path)
    print(f"Loaded {len(train_df)} training samples, {len(val_df)} validation samples")
    
    # 토크나이저
    model_name = config['model']['name']
    tokenizer = get_tokenizer(model_name)
    sep_token = get_sep_token(model_name)
    
    # 대화 맥락 설정
    context_config = get_context_config(config)
    use_context = context_config['enabled']
    if args.no_context:
        use_context = False
    
    # 전처리 (difficulty 정보 포함)
    loss_config = get_loss_config(config)
    use_soft_labels = loss_config['type'] == 'soft_bce'
    
    if use_context:
        print("\n전처리 중... (대화 맥락 포함)")
        train_df = preprocess_data_with_context(
            train_df, tokenizer, CRITERIA,
            max_length=context_config['max_length'],
            max_prev_turns=context_config['max_prev_turns'],
            sep_token=sep_token,
            include_soft_labels=True
        )
        val_df = preprocess_data_with_context(
            val_df, tokenizer, CRITERIA,
            max_length=context_config['max_length'],
            max_prev_turns=context_config['max_prev_turns'],
            sep_token=sep_token,
            include_soft_labels=True
        )
        max_length = context_config['max_length']
    else:
        print("\n전처리 중... (맥락 미포함)")
    train_df = preprocess_data(train_df, CRITERIA, sep_token, include_soft_labels=True)
    val_df = preprocess_data(val_df, CRITERIA, sep_token, include_soft_labels=True)
        max_length = config.get('tokenizer', {}).get('max_length', 128)
    
    print_label_distribution(train_df)
    
    train_dataset = CurriculumDataset(
        train_df, tokenizer, max_length, CRITERIA, use_soft_labels=use_soft_labels
    )
    
    # 검증 데이터셋은 일반 데이터셋 사용
    from src.data.dataset import QualityEvalDataset
    val_dataset = QualityEvalDataset(
        val_df, tokenizer, max_length, CRITERIA, use_soft_labels=use_soft_labels
    )
    
    print(f"\nDataset sizes: train={len(train_dataset)}, val={len(val_dataset)}")
    
    # 모델 로드
    print(f"\n모델 로드: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(CRITERIA),
        problem_type="multi_label_classification"
    )
    model.to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"파라미터 수: {param_count:,}")
    
    # 출력 디렉토리
    training_config = get_training_config(config)
    output_dir = config.get('output', {}).get('dir', './outputs')
    output_dir = Path(__file__).parent.parent / output_dir / f"curriculum-{args.strategy}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 학습 설정
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=training_config['num_epochs'],
        per_device_train_batch_size=training_config['batch_size'],
        per_device_eval_batch_size=training_config['batch_size'] * 2,
        learning_rate=training_config['learning_rate'],
        warmup_ratio=training_config['warmup_ratio'],
        weight_decay=training_config['weight_decay'],
        logging_steps=training_config['logging_steps'],
        eval_strategy=training_config['eval_strategy'],
        save_strategy=training_config['save_strategy'],
        save_total_limit=training_config['save_total_limit'],
        load_best_model_at_end=True,
        metric_for_best_model='eval_macro_f1',
        greater_is_better=True,
        fp16=training_config['fp16'],
        bf16=training_config['bf16'],
        max_grad_norm=training_config['max_grad_norm'],
        lr_scheduler_type=training_config['lr_scheduler_type'],
        optim=training_config['optim'],
        adam_beta1=training_config['adam_beta1'],
        adam_beta2=training_config['adam_beta2'],
        adam_epsilon=training_config['adam_epsilon'],
        report_to='wandb' if wandb_enabled else 'none',
        seed=training_config['seed'],
        remove_unused_columns=False,
    )
    
    # 메트릭
    metrics_computer = MetricsComputer(
        criteria=CRITERIA,
        threshold=0.5,
        optimize_threshold=True
    )
    
    # 손실 함수 설정
    loss_kwargs = {}
    if loss_config['type'] == 'label_smoothing':
        loss_kwargs = {'smoothing': loss_config['label_smoothing_alpha']}
    elif loss_config['type'] == 'focal':
        loss_kwargs = {'gamma': loss_config['focal_gamma'], 'alpha': loss_config['focal_alpha']}
    
    # Early Stopping 설정
    early_stopping_config = get_early_stopping_config(config)
    early_stopping_patience = early_stopping_config['patience'] if early_stopping_config['enabled'] else 0
    early_stopping_threshold = early_stopping_config['threshold']
    
    # Curriculum Learning 학습
    print("\n" + "=" * 50)
    print(f"Curriculum Learning 시작 (전략: {args.strategy})")
    print(f"  - Optimizer: {training_config['optim']}")
    print(f"  - LR Scheduler: {training_config['lr_scheduler_type']}")
    print(f"  - Max Grad Norm: {training_config['max_grad_norm']}")
    print(f"  - Early Stopping: patience={early_stopping_patience}, threshold={early_stopping_threshold}")
    print("=" * 50)
    
    trainer, train_result = train_with_curriculum(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
        training_args=training_args,
        compute_metrics=metrics_computer,
        curriculum_strategy=args.strategy,
        loss_type=loss_config['type'],
        loss_kwargs=loss_kwargs,
        use_soft_labels=use_soft_labels,
        early_stopping_patience=early_stopping_patience,
        early_stopping_threshold=early_stopping_threshold
    )
    
    print("\n학습 완료!")
    
    # 평가
    print("\n" + "=" * 50)
    print("최종 평가")
    print("=" * 50)
    
    eval_results = trainer.evaluate()
    
    for key, value in sorted(eval_results.items()):
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
    
    # 모델 저장
    save_path = output_dir / 'best_model'
    trainer.save_model(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    print(f"\n모델 저장: {save_path}")
    
    # wandb 종료
    if wandb_enabled:
        finish_wandb()
    
    print("\n✅ Curriculum Learning 완료!")


if __name__ == '__main__':
    main()
