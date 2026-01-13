"""
Curriculum Learning을 위한 데이터셋

샘플 난이도에 따라 학습 순서를 조절합니다.
- easy: 만장일치에 가까운 샘플 (soft label이 0.9 이상 또는 0.1 이하)
- medium: 중간 확신 샘플
- hard: 의견이 갈린 샘플 (soft label이 0.4~0.6 범위)
"""

import torch
from torch.utils.data import Dataset, Sampler
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Iterator
from transformers import PreTrainedTokenizer

from .preprocessing import CRITERIA


class CurriculumDataset(Dataset):
    """
    Curriculum Learning을 위한 데이터셋
    
    난이도 정보와 함께 데이터를 제공합니다.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 128,
        criteria: List[str] = CRITERIA,
        use_soft_labels: bool = False,
        text_column: str = 'input_text'
    ):
        """
        Args:
            df: 전처리된 데이터프레임 (difficulty 컬럼 포함)
            tokenizer: HuggingFace 토크나이저
            max_length: 최대 토큰 길이
            criteria: 평가 기준 리스트
            use_soft_labels: soft label 사용 여부
            text_column: 입력 텍스트 컬럼명
        """
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.criteria = criteria
        self.use_soft_labels = use_soft_labels
        self.text_column = text_column
        
        # 난이도 정보
        if 'difficulty' in df.columns:
            self.difficulties = df['difficulty'].values
        else:
            # difficulty 컬럼이 없으면 모두 medium으로 설정
            self.difficulties = ['medium'] * len(df)
        
        # 난이도별 인덱스
        self.easy_indices = np.where(np.array(self.difficulties) == 'easy')[0]
        self.medium_indices = np.where(np.array(self.difficulties) == 'medium')[0]
        self.hard_indices = np.where(np.array(self.difficulties) == 'hard')[0]
        
        print(f"\n[Curriculum Dataset]")
        print(f"  Easy samples: {len(self.easy_indices)} ({len(self.easy_indices)/len(df)*100:.1f}%)")
        print(f"  Medium samples: {len(self.medium_indices)} ({len(self.medium_indices)/len(df)*100:.1f}%)")
        print(f"  Hard samples: {len(self.hard_indices)} ({len(self.hard_indices)/len(df)*100:.1f}%)")
        
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        
        text = str(row[self.text_column])
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # 라벨 준비
        if self.use_soft_labels and 'soft_labels' in self.df.columns:
            labels = torch.tensor(row['soft_labels'], dtype=torch.float)
        elif 'hard_labels' in self.df.columns:
            labels = torch.tensor(row['hard_labels'], dtype=torch.float)
        else:
            labels = []
            for c in self.criteria:
                col = f'{c}_majority'
                val = row.get(col, 0)
                labels.append(float(val) if not pd.isna(val) else 0.0)
            labels = torch.tensor(labels, dtype=torch.float)
        
        result = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels,
        }
        
        # soft labels 반환
        if 'soft_labels' in self.df.columns:
            result['soft_labels'] = torch.tensor(row['soft_labels'], dtype=torch.float)
        
        # 난이도 정보
        result['difficulty'] = self.difficulties[idx]
        
        return result
    
    def get_indices_by_difficulty(self, difficulties: List[str]) -> np.ndarray:
        """특정 난이도의 인덱스 반환"""
        indices = []
        for diff in difficulties:
            if diff == 'easy':
                indices.extend(self.easy_indices.tolist())
            elif diff == 'medium':
                indices.extend(self.medium_indices.tolist())
            elif diff == 'hard':
                indices.extend(self.hard_indices.tolist())
        return np.array(indices)


class CurriculumSampler(Sampler):
    """
    Curriculum Learning을 위한 Sampler
    
    에폭이 진행됨에 따라 점점 어려운 샘플을 포함합니다.
    """
    
    def __init__(
        self,
        dataset: CurriculumDataset,
        total_epochs: int,
        current_epoch: int = 0,
        strategy: str = 'linear',  # linear, sqrt, step
        shuffle: bool = True,
        seed: int = 42
    ):
        """
        Args:
            dataset: CurriculumDataset 인스턴스
            total_epochs: 총 에폭 수
            current_epoch: 현재 에폭 (0부터 시작)
            strategy: 난이도 증가 전략
            shuffle: 샘플 셔플 여부
            seed: 랜덤 시드
        """
        self.dataset = dataset
        self.total_epochs = total_epochs
        self.current_epoch = current_epoch
        self.strategy = strategy
        self.shuffle = shuffle
        self.seed = seed
        
        self.rng = np.random.RandomState(seed)
        
    def set_epoch(self, epoch: int):
        """에폭 설정"""
        self.current_epoch = epoch
        
    def _get_difficulty_ratio(self) -> float:
        """
        현재 에폭에서 포함할 난이도 비율 계산
        
        0: easy만, 0.5: easy+medium 절반, 1: 전체
        """
        progress = self.current_epoch / max(self.total_epochs - 1, 1)
        
        if self.strategy == 'linear':
            return progress
        elif self.strategy == 'sqrt':
            return np.sqrt(progress)
        elif self.strategy == 'step':
            if progress < 0.33:
                return 0.0
            elif progress < 0.66:
                return 0.5
            else:
                return 1.0
        else:
            return progress
    
    def __iter__(self) -> Iterator[int]:
        ratio = self._get_difficulty_ratio()
        
        # 난이도에 따른 인덱스 선택
        easy_indices = self.dataset.easy_indices.tolist()
        medium_indices = self.dataset.medium_indices.tolist()
        hard_indices = self.dataset.hard_indices.tolist()
        
        # ratio에 따라 포함할 샘플 결정
        if ratio < 0.5:
            # easy + medium의 일부
            selected_indices = easy_indices.copy()
            medium_count = int(len(medium_indices) * (ratio * 2))
            selected_indices.extend(medium_indices[:medium_count])
        else:
            # easy + medium + hard의 일부
            selected_indices = easy_indices + medium_indices
            hard_count = int(len(hard_indices) * ((ratio - 0.5) * 2))
            selected_indices.extend(hard_indices[:hard_count])
        
        if self.shuffle:
            self.rng.shuffle(selected_indices)
        
        return iter(selected_indices)
    
    def __len__(self) -> int:
        ratio = self._get_difficulty_ratio()
        
        easy_count = len(self.dataset.easy_indices)
        medium_count = len(self.dataset.medium_indices)
        hard_count = len(self.dataset.hard_indices)
        
        if ratio < 0.5:
            return easy_count + int(medium_count * (ratio * 2))
        else:
            return easy_count + medium_count + int(hard_count * ((ratio - 0.5) * 2))


def compute_sample_difficulty(soft_labels: List[float]) -> float:
    """
    샘플의 난이도 점수 계산 (0: 쉬움, 1: 어려움)
    
    soft_label이 0.5에 가까울수록 어려움
    """
    difficulties = [1 - 2 * abs(sl - 0.5) for sl in soft_labels]
    return np.mean(difficulties)


def get_difficulty_from_score(score: float) -> str:
    """난이도 점수를 카테고리로 변환"""
    if score < 0.3:
        return 'easy'
    elif score < 0.6:
        return 'medium'
    else:
        return 'hard'
