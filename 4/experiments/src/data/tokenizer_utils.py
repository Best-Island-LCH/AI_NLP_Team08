"""
모델별 토크나이저 유틸리티

각 모델의 토크나이저 특성에 맞는 전처리를 수행합니다.
"""

from typing import Dict, Optional, Any
from transformers import AutoTokenizer
import yaml
from pathlib import Path


# 모델별 설정 (5가지 아키텍처 계열 + 확장)
MODEL_CONFIGS = {
    # === BERT 계열 ===
    'klue/bert-base': {
        'tokenizer': 'klue/bert-base',
        'sep_token': '[SEP]',
        'cls_token': '[CLS]',
        'pad_token': '[PAD]',
        'max_position_embeddings': 512,
        'trust_remote_code': False,
    },
    
    # === RoBERTa 계열 ===
    'klue/roberta-base': {
        'tokenizer': 'klue/roberta-base',
        'sep_token': '</s>',
        'cls_token': '<s>',
        'pad_token': '<pad>',
        'max_position_embeddings': 514,
        'trust_remote_code': False,
    },
    'klue/roberta-large': {
        'tokenizer': 'klue/roberta-large',
        'sep_token': '</s>',
        'cls_token': '<s>',
        'pad_token': '<pad>',
        'max_position_embeddings': 514,
        'trust_remote_code': False,
    },
    
    # === ELECTRA 계열 ===
    'monologg/koelectra-base-v3-discriminator': {
        'tokenizer': 'monologg/koelectra-base-v3-discriminator',
        'sep_token': '[SEP]',
        'cls_token': '[CLS]',
        'pad_token': '[PAD]',
        'max_position_embeddings': 512,
        'trust_remote_code': False,
    },
    
    # === DistilBERT 계열 ===
    'monologg/distilkobert': {
        'tokenizer': 'monologg/distilkobert',
        'sep_token': '[SEP]',
        'cls_token': '[CLS]',
        'pad_token': '[PAD]',
        'max_position_embeddings': 512,
        'trust_remote_code': True,
    },
    
    # === DeBERTa 계열 ===
    'team-lucid/deberta-v3-base-korean': {
        'tokenizer': 'team-lucid/deberta-v3-base-korean',
        'sep_token': '[SEP]',
        'cls_token': '[CLS]',
        'pad_token': '[PAD]',
        'max_position_embeddings': 512,
        'trust_remote_code': False,
    },
    
    # === Legacy (호환성) ===
    'monologg/kobert': {
        'tokenizer': 'monologg/kobert',
        'sep_token': '[SEP]',
        'cls_token': '[CLS]',
        'pad_token': '[PAD]',
        'max_position_embeddings': 512,
        'trust_remote_code': True,
    },
}


def get_tokenizer(model_name: str) -> AutoTokenizer:
    """
    모델에 맞는 토크나이저 로드
    
    Args:
        model_name: HuggingFace 모델 ID
        
    Returns:
        토크나이저 객체
    """
    config = MODEL_CONFIGS.get(model_name, {})
    trust_remote_code = config.get('trust_remote_code', False)
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code
    )
    
    return tokenizer


def get_sep_token(model_name: str) -> str:
    """모델의 SEP 토큰 반환"""
    config = MODEL_CONFIGS.get(model_name, {})
    return config.get('sep_token', '[SEP]')


def format_input_text(
    question: str,
    response: str,
    model_name: str = 'klue/roberta-base'
) -> str:
    """
    모델에 맞는 입력 텍스트 포맷
    
    Args:
        question: 질문 텍스트
        response: 응답 텍스트
        model_name: 모델 이름
        
    Returns:
        포맷된 입력 텍스트
    """
    sep_token = get_sep_token(model_name)
    
    # 결측치 처리
    question = question if question and str(question) != 'nan' else ''
    response = response if response and str(response) != 'nan' else ''
    
    return f"{question} {sep_token} {response}"


def tokenize_batch(
    texts: list,
    tokenizer: AutoTokenizer,
    max_length: int = 128,
    padding: str = 'max_length',
    truncation: bool = True,
    return_tensors: str = 'pt'
) -> Dict[str, Any]:
    """
    배치 토큰화
    
    Args:
        texts: 텍스트 리스트
        tokenizer: 토크나이저
        max_length: 최대 길이
        padding: 패딩 방식
        truncation: truncation 여부
        return_tensors: 반환 텐서 타입
        
    Returns:
        토큰화된 결과
    """
    return tokenizer(
        texts,
        max_length=max_length,
        padding=padding,
        truncation=truncation,
        return_tensors=return_tensors
    )


def get_model_config(model_name: str) -> Dict[str, Any]:
    """모델 설정 반환"""
    return MODEL_CONFIGS.get(model_name, MODEL_CONFIGS['klue/roberta-base'])


def load_model_configs(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    YAML 설정 파일에서 모델 설정 로드
    
    Args:
        config_path: 설정 파일 경로
        
    Returns:
        모델 설정 딕셔너리
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / 'config' / 'model_configs.yaml'
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
