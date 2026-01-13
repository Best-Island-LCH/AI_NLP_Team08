#!/usr/bin/env python
"""
GPU 환경변수 래퍼 스크립트

CUDA_VISIBLE_DEVICES를 subprocess 환경에서 확실하게 적용합니다.

사용법:
    python scripts/run_experiment.py 0 scripts/train.py --model klue/bert-base ...
    python scripts/run_experiment.py 1 scripts/train.py --model klue/roberta-base ...
"""

import os
import sys
import subprocess


def main():
    if len(sys.argv) < 3:
        print("Usage: python run_experiment.py <GPU_ID> <SCRIPT> [ARGS...]")
        print("Example: python run_experiment.py 0 scripts/train.py --model klue/bert-base")
        sys.exit(1)
    
    gpu_id = sys.argv[1]
    script = sys.argv[2]
    args = sys.argv[3:]
    
    # 환경변수 복사 및 GPU 설정
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = gpu_id
    
    # 로깅 (flush로 즉시 출력)
    print(f"[GPU {gpu_id}] ========================================", flush=True)
    print(f"[GPU {gpu_id}] CUDA_VISIBLE_DEVICES={gpu_id}", flush=True)
    print(f"[GPU {gpu_id}] Script: {script}", flush=True)
    print(f"[GPU {gpu_id}] Args: {' '.join(args[:6])}...", flush=True)
    print(f"[GPU {gpu_id}] ========================================", flush=True)
    
    # subprocess로 실행 (환경변수 확실히 적용)
    cmd = [sys.executable, script] + args
    result = subprocess.run(cmd, env=env)
    
    if result.returncode == 0:
        print(f"[GPU {gpu_id}] Completed successfully", flush=True)
    else:
        print(f"[GPU {gpu_id}] Failed with code {result.returncode}", flush=True)
    
    sys.exit(result.returncode)


if __name__ == '__main__':
    main()
