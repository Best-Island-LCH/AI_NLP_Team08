"""
학습 Trainer 모듈

HuggingFace Trainer를 확장하여 커스텀 기능을 추가합니다.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from typing import Dict, Optional, Any, List, Tuple
import numpy as np
from tqdm import tqdm

from .losses import get_loss_function


class QualityTrainer(Trainer):
    """
    AI 품질 평가 모델 학습용 커스텀 Trainer
    
    - Soft BCE Loss 지원
    - Label Smoothing 지원
    - Custom 평가 메트릭 지원
    """
    
    def __init__(
        self,
        loss_type: str = 'bce',
        loss_kwargs: Optional[Dict] = None,
        use_soft_labels: bool = False,
        **kwargs
    ):
        """
        Args:
            loss_type: 손실 함수 타입 ('bce', 'soft_bce', 'label_smoothing')
            loss_kwargs: 손실 함수 추가 인자
            use_soft_labels: soft label 사용 여부
            **kwargs: Trainer 기본 인자
        """
        super().__init__(**kwargs)
        
        self.loss_type = loss_type
        self.loss_kwargs = loss_kwargs or {}
        self.use_soft_labels = use_soft_labels
        
        # 손실 함수 설정
        self.custom_loss_fn = get_loss_function(loss_type, **self.loss_kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        손실 계산 오버라이드
        
        Soft label을 사용하는 경우 커스텀 손실 함수 적용
        """
        labels = inputs.pop('labels')
        soft_labels = inputs.pop('soft_labels', None)
        
        # 비텐서 필드 제거 (conversation_id 등)
        inputs.pop('conversation_id', None)
        
        # 모델 순전파
        outputs = model(**inputs)
        logits = outputs.get('logits', outputs[0] if isinstance(outputs, tuple) else outputs)
        
        # 손실 계산
        if self.loss_type == 'soft_bce' and soft_labels is not None:
            # Soft BCE Loss 사용
            loss = self.custom_loss_fn(logits, soft_labels)
        elif self.loss_type == 'uncertainty' and soft_labels is not None:
            # Uncertainty-Aware Loss 사용
            loss = self.custom_loss_fn(logits, labels, soft_labels)
        else:
            # 기본 손실 함수 사용
            loss = self.custom_loss_fn(logits, labels)
        
        return (loss, {'logits': logits}) if return_outputs else loss


def create_training_args(
    output_dir: str,
    num_epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    fp16: bool = True,
    bf16: bool = False,
    logging_steps: int = 100,
    eval_strategy: str = 'epoch',
    eval_steps: Optional[int] = None,
    save_strategy: str = 'epoch',
    save_steps: Optional[int] = None,
    save_total_limit: int = 3,
    load_best_model_at_end: bool = True,
    metric_for_best_model: str = 'macro_f1',
    greater_is_better: bool = True,
    seed: int = 42,
    report_to: str = 'wandb',
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    lr_scheduler_type: str = 'linear',
    optim: str = 'adamw_torch',
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_epsilon: float = 1e-8,
    dataloader_num_workers: int = 8,  # CPU 병렬 데이터 로딩
    dataloader_pin_memory: bool = True,  # GPU 전송 가속
    **kwargs
) -> TrainingArguments:
    """
    TrainingArguments 생성 유틸리티
    
    Args:
        output_dir: 출력 디렉토리
        num_epochs: 에폭 수
        batch_size: 배치 크기
        learning_rate: 학습률
        warmup_ratio: 워밍업 비율
        weight_decay: 가중치 감쇠
        fp16: Mixed precision (FP16) 사용
        bf16: Mixed precision (BF16) 사용 (Ampere 이상 GPU 권장)
        logging_steps: 로깅 간격
        eval_strategy: 평가 전략 ('epoch', 'steps', 'no')
        eval_steps: 평가 간격 (eval_strategy='steps' 시 사용)
        save_strategy: 저장 전략 ('epoch', 'steps', 'no')
        save_steps: 저장 간격 (save_strategy='steps' 시 사용)
        save_total_limit: 체크포인트 최대 저장 개수
        load_best_model_at_end: 학습 후 최고 모델 로드
        metric_for_best_model: 최고 모델 선택 메트릭
        greater_is_better: 메트릭이 클수록 좋은지
        seed: 랜덤 시드
        report_to: 리포팅 대상 ('wandb', 'tensorboard', 'none')
        gradient_accumulation_steps: 그래디언트 누적 스텝
        max_grad_norm: Gradient clipping (최대 gradient norm)
        lr_scheduler_type: 학습률 스케줄러 타입 ('linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant')
        optim: Optimizer 타입 ('adamw_torch', 'adamw_hf', 'adafactor', 'sgd', 'adagrad', 'adam')
        adam_beta1: Adam beta1 파라미터
        adam_beta2: Adam beta2 파라미터
        adam_epsilon: Adam epsilon 파라미터
        **kwargs: 추가 인자
        
    Returns:
        TrainingArguments
    """
    # Mixed precision 설정
    use_fp16 = fp16 and torch.cuda.is_available() and not bf16
    use_bf16 = bf16 and torch.cuda.is_available()
    
    args_dict = {
        'output_dir': output_dir,
        'num_train_epochs': num_epochs,
        'per_device_train_batch_size': batch_size,
        'per_device_eval_batch_size': batch_size * 2,
        'learning_rate': learning_rate,
        'warmup_ratio': warmup_ratio,
        'weight_decay': weight_decay,
        'fp16': use_fp16,
        'bf16': use_bf16,
        'logging_steps': logging_steps,
        'eval_strategy': eval_strategy,
        'save_strategy': save_strategy,
        'save_total_limit': save_total_limit,
        'load_best_model_at_end': load_best_model_at_end,
        'metric_for_best_model': metric_for_best_model,
        'greater_is_better': greater_is_better,
        'seed': seed,
        'report_to': report_to,
        'gradient_accumulation_steps': gradient_accumulation_steps,
        'max_grad_norm': max_grad_norm,
        'lr_scheduler_type': lr_scheduler_type,
        'optim': optim,
        'adam_beta1': adam_beta1,
        'adam_beta2': adam_beta2,
        'adam_epsilon': adam_epsilon,
        'dataloader_num_workers': dataloader_num_workers,  # CPU 병렬 로딩
        'dataloader_pin_memory': dataloader_pin_memory,  # GPU 전송 가속
        'remove_unused_columns': False,  # soft_labels 유지를 위해
    }
    
    # eval_steps/save_steps 설정 (steps 전략 사용 시)
    if eval_strategy == 'steps' and eval_steps:
        args_dict['eval_steps'] = eval_steps
    if save_strategy == 'steps' and save_steps:
        args_dict['save_steps'] = save_steps
    
    args_dict.update(kwargs)
    
    return TrainingArguments(**args_dict)


def train_model(
    model,
    train_dataset,
    val_dataset,
    tokenizer,
    training_args: TrainingArguments,
    compute_metrics,
    loss_type: str = 'bce',
    loss_kwargs: Optional[Dict] = None,
    use_soft_labels: bool = False,
    early_stopping_patience: int = 3,
    early_stopping_threshold: float = 0.001,
    callbacks: Optional[List] = None
) -> Tuple[Any, Dict]:
    """
    모델 학습 함수
    
    Args:
        model: 학습할 모델
        train_dataset: 학습 데이터셋
        val_dataset: 검증 데이터셋
        tokenizer: 토크나이저
        training_args: TrainingArguments
        compute_metrics: 메트릭 계산 함수
        loss_type: 손실 함수 타입
        loss_kwargs: 손실 함수 추가 인자
        use_soft_labels: soft label 사용 여부
        early_stopping_patience: Early stopping patience (0이면 비활성화)
        early_stopping_threshold: Early stopping 최소 개선량 (이 값 이상 개선되어야 patience 리셋)
        callbacks: 추가 콜백 리스트
        
    Returns:
        (trainer, train_result) 튜플
    """
    # 콜백 설정
    all_callbacks = callbacks or []
    if early_stopping_patience > 0:
        all_callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=early_stopping_threshold
            )
        )
    
    # Trainer 생성
    trainer = QualityTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        loss_type=loss_type,
        loss_kwargs=loss_kwargs,
        use_soft_labels=use_soft_labels,
        callbacks=all_callbacks
    )
    
    # 학습
    train_result = trainer.train()
    
    return trainer, train_result


def evaluate_model(
    trainer,
    dataset,
    return_predictions: bool = False
) -> Dict[str, float]:
    """
    모델 평가 함수
    
    Args:
        trainer: Trainer 객체
        dataset: 평가 데이터셋
        return_predictions: 예측값 반환 여부
        
    Returns:
        평가 메트릭 딕셔너리
    """
    if return_predictions:
        predictions = trainer.predict(dataset)
        return {
            'metrics': predictions.metrics,
            'predictions': predictions.predictions,
            'labels': predictions.label_ids
        }
    else:
        return trainer.evaluate(dataset)
