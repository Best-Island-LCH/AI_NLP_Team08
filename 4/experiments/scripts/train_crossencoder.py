#!/usr/bin/env python
"""
Cross-Encoder 모델 학습 스크립트

Context(질문)와 Response(응답)를 분리하여 인코딩하고
Cross-Attention으로 상호작용을 모델링합니다.

사용법:
    python scripts/train_crossencoder.py --model klue/roberta-base --loss_type soft_bce
    python scripts/train_crossencoder.py --hallucination_focus  # 환각 탐지 특화
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
    load_data, preprocess_data, preprocess_for_crossencoder,
    print_label_distribution, CRITERIA, compute_soft_labels, compute_hard_labels
)
from src.data.dataset import QualityEvalDatasetWithContext
from src.data.tokenizer_utils import get_tokenizer
from src.models.cross_encoder import CrossEncoderModel, HallucinationFocusedCrossEncoder, get_cross_encoder
from src.training.losses import SoftBCELoss, LabelSmoothingBCELoss, get_loss_function
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


class CrossEncoderTrainer(Trainer):
    """Cross-Encoder 모델용 커스텀 Trainer"""
    
    def __init__(self, loss_type='bce', loss_kwargs=None, use_soft_labels=False, **kwargs):
        super().__init__(**kwargs)
        self.loss_type = loss_type
        self.loss_kwargs = loss_kwargs or {}
        self.use_soft_labels = use_soft_labels
        
        self.loss_fn = get_loss_function(loss_type, **self.loss_kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop('labels')
        soft_labels = inputs.pop('soft_labels', None)
        
        outputs = model(**inputs)
        logits = outputs['logits']
        
        # Soft labels 사용 시
        if self.use_soft_labels and soft_labels is not None and self.loss_type == 'soft_bce':
            loss = self.loss_fn(logits, soft_labels)
        else:
            loss = self.loss_fn(logits, labels.float())
        
        return (loss, outputs) if return_outputs else loss


def collate_fn(batch):
    """Cross-Encoder용 collate 함수"""
    result = {
        'context_input_ids': torch.stack([item['context_input_ids'] for item in batch]),
        'context_attention_mask': torch.stack([item['context_attention_mask'] for item in batch]),
        'response_input_ids': torch.stack([item['response_input_ids'] for item in batch]),
        'response_attention_mask': torch.stack([item['response_attention_mask'] for item in batch]),
        'labels': torch.stack([item['labels'] for item in batch]),
    }
    
    # soft_labels가 있으면 추가
    if 'soft_labels' in batch[0]:
        result['soft_labels'] = torch.stack([item['soft_labels'] for item in batch])
    
    return result


def parse_args():
    parser = argparse.ArgumentParser(description='Cross-Encoder 모델 학습')
    parser = add_common_args(parser)
    
    # Cross-Encoder 특수 설정
    parser.add_argument('--num_cross_layers', type=int, default=2,
                        help='Cross-Attention 레이어 수')
    parser.add_argument('--share_encoder', action='store_true', default=True,
                        help='Context/Response 인코더 공유')
    parser.add_argument('--hallucination_focus', action='store_true',
                        help='환각 탐지 특화 모델 사용')
    parser.add_argument('--max_context_length', type=int, default=128)
    parser.add_argument('--max_response_length', type=int, default=128)
    
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
        run_name = args.run_name or f"crossencoder-{loss_type}-{context_tag}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        init_wandb(
            config=config,
            entity=wandb_config.get('entity', 'dhj9842-hanyang-university'),
            project=wandb_config.get('project', 'mutsa-01'),
            name=run_name,
            tags=['cross-encoder', config['model']['name'].split('/')[-1], loss_type, context_tag]
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
    
    # 대화 맥락 설정
    context_config = get_context_config(config)
    use_context = context_config['enabled']
    if args.no_context:
        use_context = False
    
    # 전처리
    loss_config = get_loss_config(config)
    use_soft_labels = loss_config['type'] == 'soft_bce'
    
    if use_context:
        # 맥락 포함 Cross-Encoder 전처리 (text_a: 맥락+질문, text_b: 응답)
        print("\n전처리 중... (대화 맥락 포함, Cross-Encoder)")
        train_df = preprocess_for_crossencoder(
            train_df, tokenizer, CRITERIA,
            max_length=context_config['max_length'],
            max_prev_turns=context_config['max_prev_turns'],
            include_soft_labels=use_soft_labels
        )
        val_df = preprocess_for_crossencoder(
            val_df, tokenizer, CRITERIA,
            max_length=context_config['max_length'],
            max_prev_turns=context_config['max_prev_turns'],
            include_soft_labels=use_soft_labels
        )
        # Dataset이 human_question/bot_response를 사용하므로 text_a/text_b를 복사
        train_df['human_question'] = train_df['text_a']
        train_df['bot_response'] = train_df['text_b']
        val_df['human_question'] = val_df['text_a']
        val_df['bot_response'] = val_df['text_b']
        # 맥락 포함 시 더 긴 context length 사용
        context_max_length = context_config['max_length'] - args.max_response_length
    else:
        # 기존 방식 (맥락 미포함)
        print("\n전처리 중... (맥락 미포함, Cross-Encoder)")
        context_max_length = args.max_context_length
    
    # soft labels 계산 (soft_bce 사용 시)
    if use_soft_labels:
        train_df['soft_labels'] = train_df.apply(lambda row: compute_soft_labels(row, CRITERIA), axis=1)
        val_df['soft_labels'] = val_df.apply(lambda row: compute_soft_labels(row, CRITERIA), axis=1)
    
    # hard labels
    train_df['hard_labels'] = train_df.apply(lambda row: compute_hard_labels(row, CRITERIA), axis=1)
    val_df['hard_labels'] = val_df.apply(lambda row: compute_hard_labels(row, CRITERIA), axis=1)
    
    # NaN 제거
    hard_label_cols = [f'{c}_majority' for c in CRITERIA]
    train_df = train_df.dropna(subset=hard_label_cols).reset_index(drop=True)
    val_df = val_df.dropna(subset=hard_label_cols).reset_index(drop=True)
    
    print_label_distribution(train_df)
    
    # 데이터셋
    train_dataset = QualityEvalDatasetWithContext(
        train_df, tokenizer,
        max_context_length=context_max_length,
        max_response_length=args.max_response_length,
        criteria=CRITERIA,
        use_soft_labels=use_soft_labels
    )
    val_dataset = QualityEvalDatasetWithContext(
        val_df, tokenizer,
        max_context_length=context_max_length,
        max_response_length=args.max_response_length,
        criteria=CRITERIA,
        use_soft_labels=use_soft_labels
    )
    
    print(f"\nDataset sizes: train={len(train_dataset)}, val={len(val_dataset)}")
    
    # 모델 로드
    print(f"\n모델 로드: Cross-Encoder ({model_name})")
    model = get_cross_encoder(
        model_name=model_name,
        num_labels=len(CRITERIA),
        num_cross_layers=args.num_cross_layers,
        share_encoder=args.share_encoder,
        use_hallucination_focus=args.hallucination_focus
    )
    model.to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"파라미터 수: {param_count:,}")
    
    # 출력 디렉토리
    training_config = get_training_config(config)
    output_dir = config.get('output', {}).get('dir', './outputs')
    output_dir = Path(__file__).parent.parent / output_dir / f"crossencoder-{loss_config['type']}"
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
        remove_unused_columns=False,  # Cross-Encoder 입력 유지
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
    callbacks = []
    if early_stopping_config['enabled']:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=early_stopping_config['patience'],
            early_stopping_threshold=early_stopping_config['threshold']
        ))
    
    # Trainer
    trainer = CrossEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=metrics_computer,
        callbacks=callbacks,
        loss_type=loss_config['type'],
        loss_kwargs=loss_kwargs,
        use_soft_labels=use_soft_labels,
        data_collator=collate_fn,
    )
    
    # 학습
    print("\n" + "=" * 50)
    print("학습 시작 (Cross-Encoder)")
    print(f"  - 대화 맥락: {'포함' if use_context else '미포함'}")
    if use_context:
        print(f"  - 최대 이전 턴: {context_config['max_prev_turns']}")
    print(f"  - Context Max Length: {context_max_length}")
    print(f"  - Response Max Length: {args.max_response_length}")
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
    
    # no_hallucination 기준 특별 출력
    if 'eval_no_hallucination_f1' in eval_results:
        print(f"\n🎯 no_hallucination F1: {eval_results['eval_no_hallucination_f1']:.4f}")
    
    # 모델 저장
    save_path = output_dir / 'best_model'
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 모델 state_dict 저장
    torch.save(model.state_dict(), save_path / 'pytorch_model.bin')
    tokenizer.save_pretrained(str(save_path))
    print(f"\n모델 저장: {save_path}")
    
    # wandb 종료
    if wandb_enabled:
        finish_wandb()
    
    print("\n✅ 학습 및 평가 완료!")


if __name__ == '__main__':
    main()
