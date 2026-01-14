#!/usr/bin/env python
"""
Contrastive Learning 학습 스크립트

Supervised Contrastive Learning을 통해 더 좋은 임베딩을 학습합니다.

사용법:
    python scripts/train_contrastive.py --lambda_contrastive 0.1
    python scripts/train_contrastive.py --lambda_contrastive 0.3 --projection_dim 512
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

from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
import wandb

from src.data.preprocessing import (
    load_data, preprocess_data, preprocess_data_with_context, 
    print_label_distribution, CRITERIA
)
from src.data.dataset import QualityEvalDataset
from src.data.tokenizer_utils import get_tokenizer, get_sep_token
from src.models.contrastive_model import ContrastiveQualityModel, ContrastiveLossWithBCE
from src.evaluation.metrics import MetricsComputer
from src.utils.wandb_utils import init_wandb, log_metrics, finish_wandb
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


class ContrastiveTrainer(Trainer):
    """Contrastive Learning용 커스텀 Trainer"""
    
    def __init__(
        self,
        lambda_contrastive: float = 0.1,
        temperature: float = 0.07,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.loss_fn = ContrastiveLossWithBCE(
            lambda_contrastive=lambda_contrastive,
            temperature=temperature
        )
        
        self._step_losses = {'bce': [], 'contrastive': []}
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop('labels')
        soft_labels = inputs.pop('soft_labels', None)
        inputs.pop('conversation_id', None)
        
        outputs = model(**inputs)
        logits = outputs['logits']
        embeddings = outputs['embeddings']
        
        # Contrastive + BCE Loss
        loss, loss_dict = self.loss_fn(logits, embeddings, labels)
        
        # 손실 기록
        self._step_losses['bce'].append(loss_dict['bce_loss'])
        self._step_losses['contrastive'].append(loss_dict['contrastive_loss'])
        
        return (loss, outputs) if return_outputs else loss
    
    def log(self, logs):
        """추가 손실 로깅"""
        if self._step_losses['bce']:
            logs['bce_loss'] = np.mean(self._step_losses['bce'][-100:])
            logs['contrastive_loss'] = np.mean(self._step_losses['contrastive'][-100:])
        
        super().log(logs)


def parse_args():
    parser = argparse.ArgumentParser(description='Contrastive Learning 학습')
    parser = add_common_args(parser)
    
    # Contrastive 설정
    parser.add_argument('--lambda_contrastive', type=float, default=0.1,
                        help='Contrastive Loss 가중치')
    parser.add_argument('--projection_dim', type=int, default=256,
                        help='Projection Head 출력 차원')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='Contrastive Loss 온도')
    
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
        context_config_for_wandb = get_context_config(config)
        use_context_for_wandb = context_config_for_wandb['enabled'] and not args.no_context
        context_tag = 'with_context' if use_context_for_wandb else 'no_context'
        run_name = args.run_name or f"contrastive-{args.lambda_contrastive}-{context_tag}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        init_wandb(
            config=config,
            entity=wandb_config.get('entity', 'dhj9842-hanyang-university'),
            project=wandb_config.get('project', 'mutsa-01'),
            name=run_name,
            tags=['contrastive', f'lambda{args.lambda_contrastive}', config['model']['name'].split('/')[-1], context_tag]
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
    
    # 전처리
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
    
    train_dataset = QualityEvalDataset(
        train_df, tokenizer, max_length, CRITERIA, use_soft_labels=False
    )
    val_dataset = QualityEvalDataset(
        val_df, tokenizer, max_length, CRITERIA, use_soft_labels=False
    )
    
    print(f"\nDataset sizes: train={len(train_dataset)}, val={len(val_dataset)}")
    
    # 모델 로드
    print(f"\n모델 로드: Contrastive Quality Model ({model_name})")
    model = ContrastiveQualityModel(
        model_name=model_name,
        num_labels=len(CRITERIA),
        projection_dim=args.projection_dim,
        temperature=args.temperature
    )
    model.to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"파라미터 수: {param_count:,}")
    print(f"Contrastive Lambda: {args.lambda_contrastive}")
    print(f"Projection Dim: {args.projection_dim}")
    
    # 출력 디렉토리
    training_config = get_training_config(config)
    output_dir = config.get('output', {}).get('dir', './outputs')
    output_dir = Path(__file__).parent.parent / output_dir / f"contrastive-{args.lambda_contrastive}"
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
    
    # Early Stopping 설정
    early_stopping_config = get_early_stopping_config(config)
    callbacks = []
    if early_stopping_config['enabled']:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=early_stopping_config['patience'],
            early_stopping_threshold=early_stopping_config['threshold']
        ))
    
    # Trainer
    trainer = ContrastiveTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=metrics_computer,
        callbacks=callbacks,
        lambda_contrastive=args.lambda_contrastive,
        temperature=args.temperature,
    )
    
    # 학습
    print("\n" + "=" * 50)
    print("Contrastive Learning 시작")
    print(f"  - Optimizer: {training_config['optim']}")
    print(f"  - LR Scheduler: {training_config['lr_scheduler_type']}")
    print(f"  - Max Grad Norm: {training_config['max_grad_norm']}")
    print(f"  - Early Stopping: patience={early_stopping_config['patience']}, threshold={early_stopping_config['threshold']}")
    print("=" * 50)
    
    trainer.train()
    
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
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path / 'pytorch_model.bin')
    tokenizer.save_pretrained(str(save_path))
    print(f"\n모델 저장: {save_path}")
    
    # wandb 종료
    if wandb_enabled:
        finish_wandb()
    
    print("\n✅ Contrastive Learning 완료!")


if __name__ == '__main__':
    main()
