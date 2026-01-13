"""
설정 유틸리티 모듈

config.yaml 로드 및 CLI 인자 병합 기능을 제공합니다.
"""

import os
import yaml
import random
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from argparse import Namespace


def load_config(config_path: str) -> Dict[str, Any]:
    """
    YAML 설정 파일 로드
    
    Args:
        config_path: 설정 파일 경로
        
    Returns:
        설정 딕셔너리
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_default_config_path() -> str:
    """기본 설정 파일 경로 반환"""
    return str(Path(__file__).parent.parent.parent / 'config' / 'config.yaml')


def merge_configs(base_config: Dict[str, Any], args: Namespace) -> Dict[str, Any]:
    """
    기본 설정과 CLI 인자 병합
    
    CLI 인자가 None이 아닌 경우 기본 설정을 덮어씁니다.
    
    Args:
        base_config: 기본 설정 딕셔너리
        args: CLI 인자 Namespace
        
    Returns:
        병합된 설정 딕셔너리
    """
    config = base_config.copy()
    
    # CLI 인자 -> config 경로 매핑
    arg_mapping = {
        # 모델
        'model': ('model', 'name'),
        
        # 학습
        'batch_size': ('training', 'batch_size'),
        'learning_rate': ('training', 'learning_rate'),
        'num_epochs': ('training', 'num_epochs'),
        'warmup_ratio': ('training', 'warmup_ratio'),
        'weight_decay': ('training', 'weight_decay'),
        'seed': ('training', 'seed'),
        'gradient_accumulation_steps': ('training', 'gradient_accumulation_steps'),
        
        # LR Scheduler
        'lr_scheduler_type': ('training', 'lr_scheduler_type'),
        
        # Optimizer
        'optim': ('training', 'optim'),
        'adam_beta1': ('training', 'adam_beta1'),
        'adam_beta2': ('training', 'adam_beta2'),
        'adam_epsilon': ('training', 'adam_epsilon'),
        
        # Gradient Clipping
        'max_grad_norm': ('training', 'max_grad_norm'),
        
        # Mixed Precision
        'precision': ('training', 'precision'),
        
        # 로깅 및 평가
        'logging_steps': ('training', 'logging_steps'),
        'eval_strategy': ('training', 'eval_strategy'),
        'eval_steps': ('training', 'eval_steps'),
        'save_strategy': ('training', 'save_strategy'),
        'save_steps': ('training', 'save_steps'),
        'save_total_limit': ('training', 'save_total_limit'),
        
        # Early Stopping
        'early_stopping_patience': ('training', 'early_stopping', 'patience'),
        'early_stopping_threshold': ('training', 'early_stopping', 'threshold'),
        
        # 토크나이저
        'max_length': ('tokenizer', 'max_length'),
        
        # 대화 맥락
        'use_context': ('context', 'enabled'),
        'max_prev_turns': ('context', 'max_prev_turns'),
        'context_max_length': ('context', 'max_length'),
        
        # 손실 함수
        'loss_type': ('loss', 'type'),
        'label_smoothing_alpha': ('loss', 'label_smoothing_alpha'),
        'focal_gamma': ('loss', 'focal_gamma'),
        'focal_alpha': ('loss', 'focal_alpha'),
        
        # wandb
        'wandb_enabled': ('wandb', 'enabled'),
        'wandb_project': ('wandb', 'project'),
        'wandb_entity': ('wandb', 'entity'),
        
        # 출력
        'output_dir': ('output', 'dir'),
    }
    
    for arg_name, config_path in arg_mapping.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            # 중첩된 경로 처리
            if len(config_path) == 2:
                section, key = config_path
                if section not in config:
                    config[section] = {}
                config[section][key] = value
            elif len(config_path) == 3:
                section, subsection, key = config_path
                if section not in config:
                    config[section] = {}
                if subsection not in config[section]:
                    config[section][subsection] = {}
                config[section][subsection][key] = value
    
    # 데이터 경로 (별도 처리)
    if hasattr(args, 'train_path') and args.train_path:
        if 'data' not in config:
            config['data'] = {}
        config['data']['train_path'] = args.train_path
    if hasattr(args, 'val_path') and args.val_path:
        if 'data' not in config:
            config['data'] = {}
        config['data']['val_path'] = args.val_path
    
    return config


def get_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    TrainingArguments에 전달할 설정 추출
    
    Args:
        config: 전체 설정 딕셔너리
        
    Returns:
        학습 설정 딕셔너리
    """
    training = config.get('training', {})
    
    # Precision 설정 파싱
    precision = training.get('precision', 'fp16')
    fp16 = precision == 'fp16'
    bf16 = precision == 'bf16'
    
    return {
        'num_epochs': training.get('num_epochs', 5),
        'batch_size': training.get('batch_size', 32),
        'learning_rate': training.get('learning_rate', 2e-5),
        'warmup_ratio': training.get('warmup_ratio', 0.1),
        'weight_decay': training.get('weight_decay', 0.01),
        'fp16': fp16,
        'bf16': bf16,
        'logging_steps': training.get('logging_steps', 100),
        'eval_strategy': training.get('eval_strategy', 'epoch'),
        'eval_steps': training.get('eval_steps', 500),
        'save_strategy': training.get('save_strategy', 'epoch'),
        'save_steps': training.get('save_steps', 500),
        'save_total_limit': training.get('save_total_limit', 3),
        'seed': training.get('seed', 42),
        'gradient_accumulation_steps': training.get('gradient_accumulation_steps', 1),
        'max_grad_norm': training.get('max_grad_norm', 1.0),
        'lr_scheduler_type': training.get('lr_scheduler_type', 'linear'),
        # Optimizer 설정
        'optim': training.get('optim', 'adamw_torch'),
        'adam_beta1': training.get('adam_beta1', 0.9),
        'adam_beta2': training.get('adam_beta2', 0.999),
        'adam_epsilon': training.get('adam_epsilon', 1e-8),
    }


def get_early_stopping_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Early Stopping 설정 추출
    
    Args:
        config: 전체 설정 딕셔너리
        
    Returns:
        Early Stopping 설정 딕셔너리
    """
    early_stopping = config.get('training', {}).get('early_stopping', {})
    
    return {
        'enabled': early_stopping.get('enabled', True),
        'patience': early_stopping.get('patience', 3),
        'threshold': early_stopping.get('threshold', 0.001),
        'metric': early_stopping.get('metric', 'eval_macro_f1'),
        'mode': early_stopping.get('mode', 'max'),
    }


def get_loss_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    손실 함수 설정 추출
    
    Args:
        config: 전체 설정 딕셔너리
        
    Returns:
        손실 함수 설정 딕셔너리
    """
    loss = config.get('loss', {})
    
    return {
        'type': loss.get('type', 'bce'),
        'label_smoothing_alpha': loss.get('label_smoothing_alpha', 0.1),
        'focal_gamma': loss.get('focal_gamma', 2.0),
        'focal_alpha': loss.get('focal_alpha', 0.25),
    }


def get_context_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    대화 맥락 설정 추출
    
    Args:
        config: 전체 설정 딕셔너리
        
    Returns:
        대화 맥락 설정 딕셔너리
    """
    context = config.get('context', {})
    
    return {
        'enabled': context.get('enabled', False),
        'max_prev_turns': context.get('max_prev_turns', 7),
        'max_length': context.get('max_length', 512),
    }


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    재현성을 위한 랜덤 시드 설정
    
    Args:
        seed: 랜덤 시드
        deterministic: cuDNN 결정적 동작 여부
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # CuBLAS workspace config for deterministic operations
            os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
            # PyTorch 1.8+ 에서 사용 가능
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass


def seed_worker(worker_id: int) -> None:
    """
    DataLoader worker 시드 설정 함수
    
    DataLoader의 worker_init_fn으로 사용합니다.
    
    Args:
        worker_id: Worker ID
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_reproducibility_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    재현성 설정 추출
    
    Args:
        config: 전체 설정 딕셔너리
        
    Returns:
        재현성 설정 딕셔너리
    """
    reproducibility = config.get('training', {}).get('reproducibility', {})
    
    return {
        'deterministic': reproducibility.get('deterministic', True),
        'dataloader_seed': reproducibility.get('dataloader_seed', True),
    }


def add_common_args(parser):
    """
    공통 CLI 인자 추가
    
    모든 학습 스크립트에서 사용하는 공통 인자들을 추가합니다.
    
    Args:
        parser: argparse.ArgumentParser
    """
    # 설정 파일
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='설정 파일 경로')
    
    # 데이터
    parser.add_argument('--train_path', type=str, default=None,
                        help='학습 데이터 경로')
    parser.add_argument('--val_path', type=str, default=None,
                        help='검증 데이터 경로')
    
    # 모델
    parser.add_argument('--model', type=str, default=None,
                        help='모델 이름 (예: klue/roberta-base)')
    
    # 학습 기본
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--num_epochs', type=int, default=None)
    parser.add_argument('--max_length', type=int, default=None)
    parser.add_argument('--warmup_ratio', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=None)
    
    # LR Scheduler
    parser.add_argument('--lr_scheduler_type', type=str, default=None,
                        choices=['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant'],
                        help='학습률 스케줄러 타입')
    
    # Optimizer
    parser.add_argument('--optim', type=str, default=None,
                        choices=['adamw_torch', 'adamw_hf', 'adafactor', 'sgd', 'adagrad', 'adam'],
                        help='Optimizer 타입')
    parser.add_argument('--adam_beta1', type=float, default=None,
                        help='Adam beta1 파라미터')
    parser.add_argument('--adam_beta2', type=float, default=None,
                        help='Adam beta2 파라미터')
    parser.add_argument('--adam_epsilon', type=float, default=None,
                        help='Adam epsilon 파라미터')
    
    # Gradient Clipping
    parser.add_argument('--max_grad_norm', type=float, default=None,
                        help='Gradient clipping 최대 norm')
    
    # Mixed Precision
    parser.add_argument('--precision', type=str, default=None,
                        choices=['fp16', 'bf16', 'fp32'],
                        help='Mixed precision 설정')
    
    # 로깅 및 평가
    parser.add_argument('--logging_steps', type=int, default=None,
                        help='로깅 간격 (steps)')
    parser.add_argument('--eval_strategy', type=str, default=None,
                        choices=['epoch', 'steps', 'no'],
                        help='평가 전략')
    parser.add_argument('--eval_steps', type=int, default=None,
                        help='평가 간격 (eval_strategy=steps 시)')
    parser.add_argument('--save_strategy', type=str, default=None,
                        choices=['epoch', 'steps', 'no'],
                        help='저장 전략')
    parser.add_argument('--save_steps', type=int, default=None,
                        help='저장 간격 (save_strategy=steps 시)')
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='체크포인트 최대 저장 개수')
    
    # Early Stopping
    parser.add_argument('--early_stopping_patience', type=int, default=None,
                        help='Early stopping patience (0이면 비활성화)')
    parser.add_argument('--early_stopping_threshold', type=float, default=None,
                        help='Early stopping 최소 개선량')
    
    # 손실 함수
    parser.add_argument('--loss_type', type=str, default=None,
                        choices=['bce', 'soft_bce', 'label_smoothing', 'uncertainty', 'focal'])
    parser.add_argument('--label_smoothing_alpha', type=float, default=None)
    parser.add_argument('--focal_gamma', type=float, default=None)
    parser.add_argument('--focal_alpha', type=float, default=None)
    
    # wandb
    parser.add_argument('--wandb_enabled', action='store_true', default=None,
                        help='wandb 활성화')
    parser.add_argument('--no_wandb', action='store_true',
                        help='wandb 비활성화')
    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--run_name', type=str, default=None)
    
    # 대화 맥락
    parser.add_argument('--use_context', action='store_true', default=None,
                        help='대화 맥락 포함 여부')
    parser.add_argument('--no_context', action='store_true',
                        help='대화 맥락 미포함 (명시적)')
    parser.add_argument('--max_prev_turns', type=int, default=None,
                        help='최대 포함할 이전 턴 수')
    parser.add_argument('--context_max_length', type=int, default=None,
                        help='맥락 포함 시 최대 토큰 길이')
    
    # 기타
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--sample_size', type=int, default=None,
                        help='데이터 샘플 크기 (디버깅용)')
    
    # GPU 지정
    parser.add_argument('--gpu', type=str, default=None,
                        help='사용할 GPU 번호 (예: 0, 1)')
    
    return parser


def setup_experiment(args: Namespace, config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    실험 설정 초기화
    
    설정 파일 로드, CLI 인자 병합, 시드 설정 등을 수행합니다.
    
    Args:
        args: CLI 인자 Namespace
        config_path: 설정 파일 경로 (None이면 args.config 사용)
        
    Returns:
        최종 설정 딕셔너리
    """
    # 설정 파일 로드
    if config_path is None:
        config_path = getattr(args, 'config', get_default_config_path())
    
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent.parent.parent / config_path
    
    if config_path.exists():
        config = load_config(str(config_path))
    else:
        config = {}
    
    # CLI 인자 병합
    config = merge_configs(config, args)
    
    # 시드 설정
    seed = config.get('training', {}).get('seed', 42)
    reproducibility = get_reproducibility_config(config)
    set_seed(seed, deterministic=reproducibility['deterministic'])
    
    return config
