"""
PyTorch Dataset 클래스

AI 품질 평가 데이터셋을 위한 Dataset 클래스를 정의합니다.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from transformers import PreTrainedTokenizer

from .preprocessing import CRITERIA


class QualityEvalDataset(Dataset):
    """
    AI 품질 평가 데이터셋
    
    Hard label과 Soft label을 모두 지원합니다.
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
            df: 전처리된 데이터프레임
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
        
        # 라벨 컬럼 확인
        self.has_soft_labels = 'soft_labels' in df.columns
        self.has_hard_labels = 'hard_labels' in df.columns
        
        # conversation_id가 있으면 저장 (평가용)
        self.has_conversation_id = 'conversation_id' in df.columns
        
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        
        # 텍스트 가져오기
        text = str(row[self.text_column])
        
        # 토큰화
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # 라벨 준비
        if self.use_soft_labels and self.has_soft_labels:
            labels = torch.tensor(row['soft_labels'], dtype=torch.float)
        elif self.has_hard_labels:
            labels = torch.tensor(row['hard_labels'], dtype=torch.float)
        else:
            # fallback: majority 컬럼에서 직접 가져오기
            labels = []
            for c in self.criteria:
                col = f'{c}_majority'
                val = row.get(col, 0)
                labels.append(float(val) if not pd.isna(val) else 0.0)
            labels = torch.tensor(labels, dtype=torch.float)
        
        result = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels
        }
        
        # Soft labels도 함께 반환 (손실 함수에서 사용)
        if self.has_soft_labels and not self.use_soft_labels:
            result['soft_labels'] = torch.tensor(row['soft_labels'], dtype=torch.float)
        
        # conversation_id 반환 (평가용)
        if self.has_conversation_id:
            result['conversation_id'] = row['conversation_id']
        
        return result


class QualityEvalDatasetWithContext(Dataset):
    """
    Cross-Encoder용 데이터셋
    
    Context(질문)와 Response(응답)를 분리하여 제공합니다.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizer,
        max_context_length: int = 128,
        max_response_length: int = 128,
        criteria: List[str] = CRITERIA,
        use_soft_labels: bool = False
    ):
        """
        Args:
            df: 전처리된 데이터프레임
            tokenizer: HuggingFace 토크나이저
            max_context_length: Context 최대 길이
            max_response_length: Response 최대 길이
            criteria: 평가 기준 리스트
            use_soft_labels: soft label 사용 여부
        """
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_context_length = max_context_length
        self.max_response_length = max_response_length
        self.criteria = criteria
        self.use_soft_labels = use_soft_labels
        
        # soft_labels 컬럼 존재 여부
        self.has_soft_labels = 'soft_labels' in df.columns
        self.has_hard_labels = 'hard_labels' in df.columns
        
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        
        # Context와 Response 분리
        context = str(row.get('human_question', ''))
        response = str(row.get('bot_response', ''))
        
        # NaN 처리
        if context == 'nan':
            context = ''
        if response == 'nan':
            response = ''
        
        # Context 토큰화
        context_encoding = self.tokenizer(
            context,
            truncation=True,
            max_length=self.max_context_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Response 토큰화
        response_encoding = self.tokenizer(
            response,
            truncation=True,
            max_length=self.max_response_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # 라벨 준비
        if self.use_soft_labels and self.has_soft_labels:
            labels = torch.tensor(row['soft_labels'], dtype=torch.float)
        elif self.has_hard_labels:
            labels = torch.tensor(row['hard_labels'], dtype=torch.float)
        else:
            labels = []
            for c in self.criteria:
                col = f'{c}_majority'
                val = row.get(col, 0)
                labels.append(float(val) if not pd.isna(val) else 0.0)
            labels = torch.tensor(labels, dtype=torch.float)
        
        result = {
            'context_input_ids': context_encoding['input_ids'].squeeze(0),
            'context_attention_mask': context_encoding['attention_mask'].squeeze(0),
            'response_input_ids': response_encoding['input_ids'].squeeze(0),
            'response_attention_mask': response_encoding['attention_mask'].squeeze(0),
            'labels': labels
        }
        
        # Soft labels도 함께 반환 (손실 함수에서 사용)
        if self.has_soft_labels and not self.use_soft_labels:
            result['soft_labels'] = torch.tensor(row['soft_labels'], dtype=torch.float)
        
        return result


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    배치 collate 함수
    
    conversation_id 등 비텐서 필드 처리
    """
    # 텐서 필드
    tensor_keys = ['input_ids', 'attention_mask', 'labels', 'soft_labels']
    
    result = {}
    
    for key in tensor_keys:
        if key in batch[0]:
            result[key] = torch.stack([item[key] for item in batch])
    
    # 비텐서 필드 (conversation_id 등)
    if 'conversation_id' in batch[0]:
        result['conversation_id'] = [item['conversation_id'] for item in batch]
    
    return result
