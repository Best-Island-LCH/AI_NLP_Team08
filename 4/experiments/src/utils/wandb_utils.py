"""
wandb 유틸리티 모듈

실험 추적을 위한 wandb 관련 함수들을 제공합니다.
"""

import os
import wandb
from typing import Dict, Optional, Any, List
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def init_wandb(
    config: Dict[str, Any],
    entity: str = "dhj9842-hanyang-university",
    project: str = "mutsa-01",
    name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    notes: Optional[str] = None,
    mode: str = "online"  # "online", "offline", "disabled"
) -> Optional[wandb.run]:
    """
    wandb 실험 초기화
    
    Args:
        config: 하이퍼파라미터 설정
        entity: wandb entity (팀 또는 사용자)
        project: wandb 프로젝트 이름
        name: run 이름 (None이면 자동 생성)
        tags: 태그 리스트
        notes: 메모
        mode: wandb 모드
        
    Returns:
        wandb run 객체
    """
    # API 키 설정 (환경 변수에서 가져오거나 직접 설정)
    api_key = os.environ.get('WANDB_API_KEY')
    if api_key:
        wandb.login(key=api_key)
    
    run = wandb.init(
        entity=entity,
        project=project,
        name=name,
        config=config,
        tags=tags,
        notes=notes,
        mode=mode,
        reinit=True
    )
    
    return run


def log_metrics(
    metrics: Dict[str, float],
    step: Optional[int] = None,
    prefix: str = ""
) -> None:
    """
    메트릭 로깅
    
    Args:
        metrics: 메트릭 딕셔너리
        step: 현재 스텝 (None이면 자동)
        prefix: 메트릭 이름 접두사 (예: "train/", "val/")
    """
    if wandb.run is None:
        return
    
    # prefix 적용
    if prefix:
        metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
    
    wandb.log(metrics, step=step)


def log_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    step: Optional[int] = None
) -> None:
    """
    Confusion Matrix 로깅
    
    Args:
        y_true: 실제 라벨
        y_pred: 예측 라벨
        class_names: 클래스 이름 리스트
        title: 제목
        step: 현재 스텝
    """
    if wandb.run is None:
        return
    
    # wandb confusion matrix
    wandb.log({
        title: wandb.plot.confusion_matrix(
            probs=None,
            y_true=y_true.tolist(),
            preds=y_pred.tolist(),
            class_names=class_names
        )
    }, step=step)


def log_table(
    data: Dict[str, List],
    table_name: str = "results"
) -> None:
    """
    테이블 데이터 로깅
    
    Args:
        data: 컬럼명 -> 값 리스트 딕셔너리
        table_name: 테이블 이름
    """
    if wandb.run is None:
        return
    
    columns = list(data.keys())
    rows = list(zip(*data.values()))
    
    table = wandb.Table(columns=columns, data=rows)
    wandb.log({table_name: table})


def log_figure(
    figure: plt.Figure,
    name: str = "plot",
    step: Optional[int] = None
) -> None:
    """
    matplotlib Figure 로깅
    
    Args:
        figure: matplotlib Figure
        name: 이름
        step: 현재 스텝
    """
    if wandb.run is None:
        return
    
    wandb.log({name: wandb.Image(figure)}, step=step)
    plt.close(figure)


def log_model(
    model_path: str,
    name: str = "model",
    aliases: Optional[List[str]] = None
) -> None:
    """
    모델 아티팩트 로깅
    
    Args:
        model_path: 모델 저장 경로
        name: 아티팩트 이름
        aliases: 별칭 리스트
    """
    if wandb.run is None:
        return
    
    artifact = wandb.Artifact(name, type='model')
    artifact.add_dir(model_path)
    wandb.log_artifact(artifact, aliases=aliases)


def finish_wandb() -> None:
    """wandb 실험 종료"""
    if wandb.run is not None:
        wandb.finish()


def create_sweep(
    sweep_config: Dict[str, Any],
    entity: str = "dhj9842-hanyang-university",
    project: str = "mutsa-01"
) -> str:
    """
    wandb Sweep 생성
    
    Args:
        sweep_config: sweep 설정
        entity: wandb entity
        project: wandb 프로젝트
        
    Returns:
        sweep_id
    """
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        entity=entity,
        project=project
    )
    
    return sweep_id


def run_sweep_agent(
    sweep_id: str,
    train_func,
    count: int = 30,
    entity: str = "dhj9842-hanyang-university",
    project: str = "mutsa-01"
) -> None:
    """
    wandb Sweep Agent 실행
    
    Args:
        sweep_id: sweep ID
        train_func: 학습 함수
        count: 실행할 run 수
        entity: wandb entity
        project: wandb 프로젝트
    """
    wandb.agent(
        sweep_id,
        function=train_func,
        count=count,
        entity=entity,
        project=project
    )


class WandbCallback:
    """
    학습 중 wandb 로깅을 위한 콜백
    """
    
    def __init__(
        self,
        log_interval: int = 100,
        log_model: bool = True
    ):
        """
        Args:
            log_interval: 로깅 간격 (스텝)
            log_model: 모델 저장 여부
        """
        self.log_interval = log_interval
        self.log_model = log_model
        self.step = 0
    
    def on_train_step(self, loss: float, learning_rate: float):
        """학습 스텝 콜백"""
        self.step += 1
        
        if self.step % self.log_interval == 0:
            log_metrics({
                'loss': loss,
                'learning_rate': learning_rate
            }, step=self.step, prefix='train/')
    
    def on_eval_end(self, metrics: Dict[str, float], epoch: int):
        """평가 종료 콜백"""
        log_metrics(metrics, prefix='val/')
    
    def on_train_end(self, model_path: Optional[str] = None):
        """학습 종료 콜백"""
        if self.log_model and model_path:
            log_model(model_path, name='final_model', aliases=['latest'])
