#!/usr/bin/env python
"""
2-GPU 병렬 실험 실행기
- GPU 0: noctx 실험 / Loss 실험 (짝수)
- GPU 1: ctx 실험 / Loss 실험 (홀수)
- Phase 간 자동 의존성: Phase 1 완료 후 Best Model 자동 선정
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime

# 프로젝트 루트 설정
ROOT = Path(__file__).parent.parent
os.chdir(ROOT)

# 색상 코드
GREEN = '\033[0;32m'
BLUE = '\033[0;34m'
CYAN = '\033[0;36m'
RED = '\033[0;31m'
NC = '\033[0m'

# 설정
PROGRESS_FILE = ROOT / "outputs" / ".progress"
RESULTS_DIR = ROOT / "outputs" / "results"
LOGS_DIR = ROOT / "logs"

def log_info(msg):
    print(f"{BLUE}[{datetime.now().strftime('%H:%M:%S')}]{NC} {msg}")

def log_ok(msg):
    print(f"{GREEN}[{datetime.now().strftime('%H:%M:%S')}] ✓{NC} {msg}")

def log_error(msg):
    print(f"{RED}[{datetime.now().strftime('%H:%M:%S')}] ✗{NC} {msg}")

def log_phase(msg):
    print(f"\n{CYAN}{'═' * 15} {msg} {'═' * 15}{NC}\n")

def is_done(name):
    """실험이 이미 완료되었는지 확인"""
    if not PROGRESS_FILE.exists():
        return False
    with open(PROGRESS_FILE) as f:
        return any(line.startswith(f"{name}:done") for line in f)

def mark_done(name):
    """실험 완료 기록"""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, 'a') as f:
        f.write(f"{name}:done:{datetime.now()}\n")
    log_ok(f"{name} 완료")

def run_on_gpu(gpu_id, script, args, name):
    """특정 GPU에서 실험 실행 (subprocess)"""
    if is_done(name):
        log_info(f"{name} 이미 완료, 스킵")
        return None
    
    log_info(f"{name} 시작 (GPU {gpu_id})...")
    
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    log_file = LOGS_DIR / f"{name}.log"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    cmd = ['python', script] + args + ['--run_name', name]
    
    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=str(ROOT)
        )
    
    return process

def wait_and_mark(process, name):
    """프로세스 완료 대기 및 기록"""
    if process is None:
        return True
    
    returncode = process.wait()
    if returncode == 0:
        mark_done(name)
        return True
    else:
        log_error(f"{name} 실패 (exit code: {returncode})")
        return False

def run_parallel_pair(gpu0_task, gpu1_task):
    """두 개의 실험을 병렬로 실행"""
    p0 = run_on_gpu(0, *gpu0_task) if gpu0_task else None
    p1 = run_on_gpu(1, *gpu1_task) if gpu1_task else None
    
    success0 = wait_and_mark(p0, gpu0_task[2]) if gpu0_task else True
    success1 = wait_and_mark(p1, gpu1_task[2]) if gpu1_task else True
    
    return success0 and success1

# ================================================
# Phase 1: 아키텍처 비교 (10 실험)
# ================================================
def run_phase1():
    log_phase("PHASE 1: Architecture Comparison")
    
    models = [
        ("klue/bert-base", "bert", "2e-5", "64", "32"),
        ("klue/roberta-base", "roberta", "2e-5", "64", "32"),
        ("monologg/koelectra-base-v3-discriminator", "electra", "3e-5", "64", "32"),
        ("monologg/distilkobert", "distilbert", "5e-5", "128", "64"),
        ("team-lucid/deberta-v3-base-korean", "deberta", "1e-5", "32", "16"),
    ]
    
    for model_id, model_name, lr, batch_noctx, batch_ctx in models:
        noctx_name = f"{model_name}-noctx"
        ctx_name = f"{model_name}-ctx"
        
        # GPU 0: noctx, GPU 1: ctx 병렬 실행
        noctx_task = (
            "scripts/train.py",
            ["--model", model_id, "--loss_type", "bce", "--learning_rate", lr,
             "--batch_size", batch_noctx, "--num_epochs", "3", "--no_context", "--max_length", "128"],
            noctx_name
        )
        
        ctx_task = (
            "scripts/train.py",
            ["--model", model_id, "--loss_type", "bce", "--learning_rate", lr,
             "--batch_size", batch_ctx, "--num_epochs", "3", "--use_context", "--max_length", "512"],
            ctx_name
        )
        
        if not run_parallel_pair(noctx_task, ctx_task):
            log_error(f"{model_name} 실험 실패")
            return False
    
    log_ok("Phase 1 완료!")
    return True

# ================================================
# Phase 1 결과 분석 및 Best Model 선정
# ================================================
def get_best_model_from_phase1():
    """Phase 1 결과에서 Best Model 선정 (Macro F1 기준)"""
    log_info("Phase 1 결과 분석 중...")
    
    results = []
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # ctx 실험 결과만 비교 (실제 사용 시나리오)
    ctx_experiments = ["bert-ctx", "roberta-ctx", "electra-ctx", "distilbert-ctx", "deberta-ctx"]
    
    model_map = {
        "bert-ctx": "klue/bert-base",
        "roberta-ctx": "klue/roberta-base",
        "electra-ctx": "monologg/koelectra-base-v3-discriminator",
        "distilbert-ctx": "monologg/distilkobert",
        "deberta-ctx": "team-lucid/deberta-v3-base-korean",
    }
    
    for exp_name in ctx_experiments:
        result_file = RESULTS_DIR / f"{exp_name}_results.json"
        if result_file.exists():
            with open(result_file) as f:
                data = json.load(f)
                results.append({
                    "name": exp_name,
                    "model": model_map[exp_name],
                    "macro_f1": data.get("eval_macro_f1", data.get("macro_f1", 0)),
                    "eval_loss": data.get("eval_loss", float('inf'))
                })
                log_info(f"  {exp_name}: macro_f1={results[-1]['macro_f1']:.4f}")
    
    if not results:
        log_error("Phase 1 결과를 찾을 수 없음. 기본값 RoBERTa 사용")
        return "klue/roberta-base", "2e-5"
    
    # Macro F1 내림차순, eval_loss 오름차순 정렬
    results.sort(key=lambda x: (-x['macro_f1'], x['eval_loss']))
    
    best = results[0]
    log_ok(f"Best Model: {best['name']} (macro_f1={best['macro_f1']:.4f})")
    
    # Learning rate 매핑
    lr_map = {
        "klue/bert-base": "2e-5",
        "klue/roberta-base": "2e-5",
        "monologg/koelectra-base-v3-discriminator": "3e-5",
        "monologg/distilkobert": "5e-5",
        "team-lucid/deberta-v3-base-korean": "1e-5",
    }
    
    return best['model'], lr_map.get(best['model'], "2e-5")

# ================================================
# Phase 2: Loss 함수 비교 (Best Model × 4 Loss)
# ================================================
def run_phase2(best_model, best_lr):
    log_phase(f"PHASE 2: Loss Function Study ({best_model.split('/')[-1]})")
    
    # BCE는 Phase 1에서 이미 수행됨 (중복 제거)
    losses = [
        ("soft_bce", "loss-softbce"),
        ("focal", "loss-focal"),
        ("asl", "loss-asl"),
        ("criterion_weighted", "loss-weighted"),
    ]
    
    # 2개씩 병렬 실행
    for i in range(0, len(losses), 2):
        tasks = []
        for j in range(2):
            if i + j < len(losses):
                loss_type, name = losses[i + j]
                tasks.append((
                    "scripts/train.py",
                    ["--model", best_model, "--loss_type", loss_type, "--learning_rate", best_lr,
                     "--batch_size", "32", "--num_epochs", "3", "--use_context", "--max_length", "512"],
                    name
                ))
        
        gpu0_task = tasks[0] if len(tasks) > 0 else None
        gpu1_task = tasks[1] if len(tasks) > 1 else None
        
        if not run_parallel_pair(gpu0_task, gpu1_task):
            return False
    
    log_ok("Phase 2 완료!")
    return True

# ================================================
# Phase 3: 고급 아키텍처 (병렬)
# ================================================
def run_phase3(best_model, best_lr):
    log_phase(f"PHASE 3: Advanced Architecture ({best_model.split('/')[-1]})")
    
    multihead_task = (
        "scripts/train_multihead.py",
        ["--model", best_model, "--loss_type", "soft_bce", "--learning_rate", best_lr,
         "--batch_size", "24", "--num_epochs", "3", "--use_context", "--max_length", "512"],
        "arch-multihead"
    )
    
    crossenc_task = (
        "scripts/train_crossencoder.py",
        ["--model", best_model, "--loss_type", "soft_bce", "--learning_rate", best_lr,
         "--batch_size", "16", "--num_epochs", "3", "--use_context", "--max_length", "512"],
        "arch-crossenc"
    )
    
    if not run_parallel_pair(multihead_task, crossenc_task):
        return False
    
    log_ok("Phase 3 완료!")
    return True

# ================================================
# Phase 4: 학습 전략 (순차)
# ================================================
def run_phase4(best_model, best_lr):
    log_phase(f"PHASE 4: Learning Strategy ({best_model.split('/')[-1]})")
    
    experiments = [
        ("scripts/train_curriculum.py",
         ["--model", best_model, "--loss_type", "soft_bce", "--learning_rate", best_lr,
          "--batch_size", "32", "--num_epochs", "5", "--use_context", "--max_length", "512", "--strategy", "sqrt"],
         "strat-curriculum"),
        ("scripts/train_contrastive.py",
         ["--model", best_model, "--learning_rate", best_lr,
          "--batch_size", "32", "--num_epochs", "3", "--use_context", "--max_length", "512",
          "--lambda_contrastive", "0.1", "--projection_dim", "256"],
         "strat-contrastive"),
    ]
    
    # 순차 실행 (메모리 많이 사용)
    for script, args, name in experiments:
        p = run_on_gpu(0, script, args, name)
        if not wait_and_mark(p, name):
            return False
    
    log_ok("Phase 4 완료!")
    return True

# ================================================
# 메인
# ================================================
def main():
    log_phase("mutsa-v2 Experiment (2-GPU Parallel)")
    log_info("환경: Threadripper PRO 7975WX + 2× RTX 3090")
    
    os.system("nvidia-smi --query-gpu=index,memory.used --format=csv")
    
    start_time = time.time()
    
    # Phase 1: 아키텍처 비교
    if not run_phase1():
        log_error("Phase 1 실패")
        return 1
    
    # Best Model 선정
    best_model, best_lr = get_best_model_from_phase1()
    
    # Phase 2: Loss 함수 비교
    if not run_phase2(best_model, best_lr):
        log_error("Phase 2 실패")
        return 1
    
    # Phase 3: 고급 아키텍처
    if not run_phase3(best_model, best_lr):
        log_error("Phase 3 실패")
        return 1
    
    # Phase 4: 학습 전략
    if not run_phase4(best_model, best_lr):
        log_error("Phase 4 실패")
        return 1
    
    elapsed = int((time.time() - start_time) / 60)
    
    log_phase("COMPLETE")
    print(f"📊 wandb: https://wandb.ai/dhj9842-hanyang-university/mutsa-v2")
    log_ok(f"총 소요: {elapsed}분")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
