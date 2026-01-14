#!/usr/bin/env python
"""
AI 품질 평가 모델 학습 스크립트

사용법:
    python scripts/train.py --config config/config.yaml
    python scripts/train.py --model klue/roberta-base --loss_type soft_bce
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

import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import wandb

from src.data.preprocessing import (
    load_data, preprocess_data, preprocess_data_with_context, 
    print_label_distribution, CRITERIA
)
from src.data.dataset import QualityEvalDataset, collate_fn
from src.data.tokenizer_utils import get_tokenizer, get_sep_token
from src.training.trainer import create_training_args, train_model
from src.training.losses import get_loss_function
from src.evaluation.metrics import MetricsComputer, compute_metrics
from src.evaluation.analysis import generate_evaluation_report
from src.utils.wandb_utils import init_wandb, log_metrics, finish_wandb
from src.utils.config_utils import (
    load_config,
    merge_configs,
    get_training_config,
    get_early_stopping_config,
    get_loss_config,
    get_context_config,
    set_seed,
    seed_worker,
    get_reproducibility_config,
    add_common_args,
    setup_experiment,
)


def parse_args():
    parser = argparse.ArgumentParser(description='AI 품질 평가 모델 학습')
    parser = add_common_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 설정 초기화 (설정 파일 로드 + CLI 병합 + 시드 설정)
    config = setup_experiment(args)
    
    # 디바이스 설정 및 메모리 최적화
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # GPU 메모리 캐시 정리
    print(f"Using device: {device}")
    
    # wandb 초기화
    wandb_config = config.get('wandb', {})
    wandb_enabled = wandb_config.get('enabled', True)
    
    # CLI 인자로 wandb 비활성화
    if args.no_wandb:
        wandb_enabled = False
    
    if wandb_enabled:
        # 맥락 설정 확인 (wandb 초기화 전에도 확인)
        context_config_for_wandb = get_context_config(config)
        use_context_for_wandb = context_config_for_wandb['enabled'] and not args.no_context
        
        context_tag = 'with_context' if use_context_for_wandb else 'no_context'
        run_name = args.run_name or f"{config['model']['name'].split('/')[-1]}_{context_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        init_wandb(
            config=config,
            entity=wandb_config.get('entity', 'dhj9842-hanyang-university'),
            project=wandb_config.get('project', 'mutsa-01'),
            name=run_name,
            tags=[config['model']['name'], config['loss']['type'], context_tag]
        )
    
    # 데이터 로드
    data_config = config.get('data', {})
    data_dir = Path(__file__).parent.parent.parent / 'data'
    
    train_path = data_config.get('train_path', str(data_dir / 'train' / 'training_all_aggregated.csv'))
    val_path = data_config.get('val_path', str(data_dir / 'val' / 'validation_all_aggregated.csv'))
    
    # 상대 경로 처리
    if not Path(train_path).is_absolute():
        train_path = str(Path(__file__).parent.parent / train_path)
    if not Path(val_path).is_absolute():
        val_path = str(Path(__file__).parent.parent / val_path)
    
    train_df, val_df = load_data(train_path, val_path)
    
    # 샘플링 (디버깅용)
    if args.sample_size:
        train_df = train_df.head(args.sample_size)
        val_df = val_df.head(args.sample_size // 5)
        print(f"Sampled: {len(train_df)} train, {len(val_df)} val")
    
    # 토크나이저 로드
    model_name = config['model']['name']
    tokenizer = get_tokenizer(model_name)
    sep_token = get_sep_token(model_name)
    
    # 대화 맥락 설정
    context_config = get_context_config(config)
    use_context = context_config['enabled']
    
    # CLI에서 명시적으로 비활성화한 경우
    if args.no_context:
        use_context = False
    
    # 데이터 전처리
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
        tokenizer_config = config.get('tokenizer', {})
        max_length = tokenizer_config.get('max_length', 128)
    
    print_label_distribution(train_df)
    
    loss_config = get_loss_config(config)
    use_soft_labels = loss_config['type'] == 'soft_bce'
    
    train_dataset = QualityEvalDataset(
        train_df, tokenizer, max_length, CRITERIA, use_soft_labels=use_soft_labels
    )
    val_dataset = QualityEvalDataset(
        val_df, tokenizer, max_length, CRITERIA, use_soft_labels=use_soft_labels
    )
    
    print(f"\nDataset sizes: train={len(train_dataset)}, val={len(val_dataset)}")
    
    # 모델 로드
    print(f"\n모델 로드: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=config['model'].get('num_labels', 9),
        problem_type="multi_label_classification"
    )
    model.to(device)
    
    # torch.compile()은 일부 환경에서 불안정하여 비활성화
    # if hasattr(torch, 'compile') and torch.cuda.is_available():
    #     model = torch.compile(model, mode='reduce-overhead')
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"파라미터 수: {param_count:,}")
    
    # 학습 설정
    training_config = get_training_config(config)
    output_config = config.get('output', {})
    
    output_dir = output_config.get('dir', './outputs')
    output_dir = Path(__file__).parent.parent / output_dir / model_name.split('/')[-1]
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = create_training_args(
        output_dir=str(output_dir),
        num_epochs=training_config['num_epochs'],
        batch_size=training_config['batch_size'],
        learning_rate=training_config['learning_rate'],
        warmup_ratio=training_config['warmup_ratio'],
        weight_decay=training_config['weight_decay'],
        fp16=training_config['fp16'],
        bf16=training_config['bf16'],
        logging_steps=training_config['logging_steps'],
        eval_strategy=training_config['eval_strategy'],
        eval_steps=training_config['eval_steps'],
        save_strategy=training_config['save_strategy'],
        save_steps=training_config['save_steps'],
        save_total_limit=training_config['save_total_limit'],
        seed=training_config['seed'],
        report_to='wandb' if wandb_enabled else 'none',
        gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
        max_grad_norm=training_config['max_grad_norm'],
        lr_scheduler_type=training_config['lr_scheduler_type'],
        optim=training_config['optim'],
        adam_beta1=training_config['adam_beta1'],
        adam_beta2=training_config['adam_beta2'],
        adam_epsilon=training_config['adam_epsilon'],
    )
    
    # 메트릭 계산 함수
    metrics_computer = MetricsComputer(
        criteria=CRITERIA,
        threshold=config.get('evaluation', {}).get('threshold', 0.5),
        optimize_threshold=config.get('evaluation', {}).get('optimize_threshold', False)
    )
    
    # 손실 함수 설정
    loss_type = loss_config['type']
    loss_kwargs = {}
    if loss_type == 'label_smoothing':
        loss_kwargs['smoothing'] = loss_config['label_smoothing_alpha']
    elif loss_type == 'focal':
        loss_kwargs['gamma'] = loss_config['focal_gamma']
        loss_kwargs['alpha'] = loss_config['focal_alpha']
    
    # Early Stopping 설정
    early_stopping_config = get_early_stopping_config(config)
    early_stopping_patience = early_stopping_config['patience'] if early_stopping_config['enabled'] else 0
    early_stopping_threshold = early_stopping_config['threshold']
    
    # 학습 시작
    print("\n" + "=" * 50)
    print("학습 시작")
    print(f"  - 대화 맥락: {'포함' if use_context else '미포함'}")
    if use_context:
        print(f"  - 최대 이전 턴: {context_config['max_prev_turns']}")
    print(f"  - Max Length: {max_length}")
    print(f"  - Optimizer: {training_config['optim']}")
    print(f"  - LR Scheduler: {training_config['lr_scheduler_type']}")
    print(f"  - Max Grad Norm: {training_config['max_grad_norm']}")
    print(f"  - Early Stopping: patience={early_stopping_patience}, threshold={early_stopping_threshold}")
    print("=" * 50)
    
    trainer, train_result = train_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
        training_args=training_args,
        compute_metrics=metrics_computer,
        loss_type=loss_type,
        loss_kwargs=loss_kwargs,
        use_soft_labels=use_soft_labels,
        early_stopping_patience=early_stopping_patience,
        early_stopping_threshold=early_stopping_threshold,
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
    if output_config.get('save_best_model', True):
        save_path = output_dir / 'best_model'
        trainer.save_model(str(save_path))
        tokenizer.save_pretrained(str(save_path))
        print(f"\n모델 저장: {save_path}")
    
    # 상세 평가 리포트 생성
    print("\n상세 평가 리포트 생성 중...")
    predictions = trainer.predict(val_dataset)
    
    report = generate_evaluation_report(
        predictions=1 / (1 + np.exp(-predictions.predictions)),  # Sigmoid
        labels=predictions.label_ids,
        conversation_ids=val_df['conversation_id'].tolist() if 'conversation_id' in val_df.columns else None,
        criteria=CRITERIA
    )
    
    # 리포트 저장
    report_path = output_dir / 'evaluation_report.json'
    
    # numpy 배열을 리스트로 변환
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
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"평가 리포트 저장: {report_path}")
    
    # wandb 종료
    if wandb_enabled:
        # 최종 메트릭 로깅
        log_metrics(eval_results, prefix='final/')
        finish_wandb()
    
    print("\n✅ 학습 및 평가 완료!")


if __name__ == '__main__':
    main()
