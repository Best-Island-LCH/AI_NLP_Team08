"""
Curriculum Learning Trainer

에폭이 진행됨에 따라 점점 어려운 샘플을 포함하는 학습 전략
"""

import torch
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Callable
import numpy as np

from ..data.curriculum_dataset import CurriculumDataset, CurriculumSampler
from .losses import get_loss_function


class CurriculumTrainer(Trainer):
    """
    Curriculum Learning을 지원하는 커스텀 Trainer
    
    에폭마다 CurriculumSampler를 업데이트하여
    점진적으로 어려운 샘플을 포함합니다.
    """
    
    def __init__(
        self,
        curriculum_strategy: str = 'linear',
        loss_type: str = 'bce',
        loss_kwargs: Optional[Dict] = None,
        use_soft_labels: bool = False,
        **kwargs
    ):
        """
        Args:
            curriculum_strategy: 'linear', 'sqrt', 'step' 중 하나
            loss_type: 손실 함수 타입
            loss_kwargs: 손실 함수 추가 인자
            use_soft_labels: soft label 사용 여부
            **kwargs: Trainer 기본 인자
        """
        super().__init__(**kwargs)
        
        self.curriculum_strategy = curriculum_strategy
        self.loss_type = loss_type
        self.loss_kwargs = loss_kwargs or {}
        self.use_soft_labels = use_soft_labels
        
        self.loss_fn = get_loss_function(loss_type, **self.loss_kwargs)
        
        # 현재 에폭 추적
        self._current_epoch = 0
    
    def get_train_dataloader(self) -> DataLoader:
        """
        Curriculum Sampler를 사용하는 DataLoader 반환
        """
        train_dataset = self.train_dataset
        
        if isinstance(train_dataset, CurriculumDataset):
            # CurriculumSampler 생성
            sampler = CurriculumSampler(
                dataset=train_dataset,
                total_epochs=int(self.args.num_train_epochs),
                current_epoch=self._current_epoch,
                strategy=self.curriculum_strategy,
                shuffle=True,
                seed=self.args.seed
            )
            
            return DataLoader(
                train_dataset,
                batch_size=self._train_batch_size,
                sampler=sampler,
                collate_fn=self.data_collator,
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )
        else:
            # 일반 Dataset인 경우 기본 DataLoader 사용
            return super().get_train_dataloader()
    
    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval):
        """에폭 종료 시 콜백 (CurriculumSampler 업데이트)"""
        # 에폭 업데이트
        if epoch is not None and epoch != self._current_epoch:
            self._current_epoch = int(epoch)
            
            # 샘플 수 로깅
            if isinstance(self.train_dataset, CurriculumDataset):
                sampler = CurriculumSampler(
                    dataset=self.train_dataset,
                    total_epochs=int(self.args.num_train_epochs),
                    current_epoch=self._current_epoch,
                    strategy=self.curriculum_strategy
                )
                print(f"\n[Curriculum] Epoch {self._current_epoch}: {len(sampler)} samples")
        
        return super()._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval)
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """손실 계산"""
        labels = inputs.pop('labels')
        soft_labels = inputs.pop('soft_labels', None)
        
        # 비텐서 필드 제거
        inputs.pop('difficulty', None)
        inputs.pop('conversation_id', None)
        
        outputs = model(**inputs)
        logits = outputs.get('logits', outputs[0] if isinstance(outputs, tuple) else outputs)
        
        # 손실 계산
        if self.loss_type == 'soft_bce' and soft_labels is not None:
            loss = self.loss_fn(logits, soft_labels)
        else:
            loss = self.loss_fn(logits, labels.float())
        
        return (loss, {'logits': logits}) if return_outputs else loss


def collate_fn_curriculum(batch):
    """Curriculum Dataset용 collate 함수"""
    result = {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'labels': torch.stack([item['labels'] for item in batch]),
    }
    
    if 'soft_labels' in batch[0]:
        result['soft_labels'] = torch.stack([item['soft_labels'] for item in batch])
    
    # difficulty는 문자열이므로 리스트로 반환
    if 'difficulty' in batch[0]:
        result['difficulty'] = [item['difficulty'] for item in batch]
    
    return result


def train_with_curriculum(
    model,
    train_dataset: CurriculumDataset,
    val_dataset,
    tokenizer,
    training_args: TrainingArguments,
    compute_metrics: Callable,
    curriculum_strategy: str = 'linear',
    loss_type: str = 'bce',
    loss_kwargs: Optional[Dict] = None,
    use_soft_labels: bool = False,
    early_stopping_patience: int = 3,
    early_stopping_threshold: float = 0.001
):
    """
    Curriculum Learning으로 모델 학습
    
    Args:
        model: 학습할 모델
        train_dataset: CurriculumDataset 인스턴스
        val_dataset: 검증 데이터셋
        tokenizer: 토크나이저
        training_args: TrainingArguments
        compute_metrics: 메트릭 계산 함수
        curriculum_strategy: 난이도 증가 전략
        loss_type: 손실 함수 타입
        loss_kwargs: 손실 함수 추가 인자
        use_soft_labels: soft label 사용 여부
        early_stopping_patience: Early stopping patience (0이면 비활성화)
        early_stopping_threshold: Early stopping 최소 개선량
        
    Returns:
        (trainer, train_result) 튜플
    """
    # 콜백 설정
    callbacks = []
    if early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=early_stopping_threshold
            )
        )
    
    # Trainer 생성
    trainer = CurriculumTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        curriculum_strategy=curriculum_strategy,
        loss_type=loss_type,
        loss_kwargs=loss_kwargs,
        use_soft_labels=use_soft_labels,
        callbacks=callbacks,
        data_collator=collate_fn_curriculum,
    )
    
    # 학습
    train_result = trainer.train()
    
    return trainer, train_result
