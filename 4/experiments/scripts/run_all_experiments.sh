#!/bin/bash
# ================================================
# Korean BERT Experiment Runner
# Project: mutsa-seq
# 순차 실행 (GPU 0 사용)
# ================================================

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1"; }
log_ok() { echo -e "${GREEN}[$(date '+%H:%M:%S')] ✓${NC} $1"; }
log_phase() { echo -e "\n${CYAN}═══════════ $1 ═══════════${NC}\n"; }

cd "$(dirname "$0")/.."
ROOT=$(pwd)
PROGRESS="$ROOT/outputs/.progress"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
export WANDB_API_KEY="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"
export CUDA_VISIBLE_DEVICES=0
mkdir -p outputs/results outputs/figures logs

is_done() { [ -f "$PROGRESS" ] && grep -q "^$1:done" "$PROGRESS"; }
mark_done() { echo "$1:done:$(date)" >> "$PROGRESS"; log_ok "$1 완료"; }

run_exp() {
    local name=$1
    shift
    if is_done "$name"; then
        log_info "$name 이미 완료, 스킵"
        return 0
    fi
    log_info "$name 시작..."
    python "$@" --run_name "$name" > "logs/${name}.log" 2>&1
    mark_done "$name"
}

# ================================================
# Phase 1: 아키텍처 비교 (5 모델)
# ================================================
run_phase1() {
    log_phase "PHASE 1: Architecture Comparison"
    
    run_exp "bert-noctx" scripts/train.py --model klue/bert-base --loss_type bce --learning_rate 2e-5 --batch_size 64 --num_epochs 3 --no_context --max_length 128
    run_exp "bert-ctx" scripts/train.py --model klue/bert-base --loss_type bce --learning_rate 2e-5 --batch_size 32 --num_epochs 3 --use_context --max_length 512
    
    run_exp "roberta-noctx" scripts/train.py --model klue/roberta-base --loss_type bce --learning_rate 2e-5 --batch_size 64 --num_epochs 3 --no_context --max_length 128
    run_exp "roberta-ctx" scripts/train.py --model klue/roberta-base --loss_type bce --learning_rate 2e-5 --batch_size 32 --num_epochs 3 --use_context --max_length 512
    
    run_exp "electra-noctx" scripts/train.py --model monologg/koelectra-base-v3-discriminator --loss_type bce --learning_rate 3e-5 --batch_size 64 --num_epochs 3 --no_context --max_length 128
    run_exp "electra-ctx" scripts/train.py --model monologg/koelectra-base-v3-discriminator --loss_type bce --learning_rate 3e-5 --batch_size 32 --num_epochs 3 --use_context --max_length 512
    
    run_exp "distilbert-noctx" scripts/train.py --model monologg/distilkobert --loss_type bce --learning_rate 5e-5 --batch_size 128 --num_epochs 3 --no_context --max_length 128
    run_exp "distilbert-ctx" scripts/train.py --model monologg/distilkobert --loss_type bce --learning_rate 5e-5 --batch_size 64 --num_epochs 3 --use_context --max_length 512
    
    run_exp "deberta-noctx" scripts/train.py --model team-lucid/deberta-v3-base-korean --loss_type bce --learning_rate 1e-5 --batch_size 32 --num_epochs 3 --no_context --max_length 128
    run_exp "deberta-ctx" scripts/train.py --model team-lucid/deberta-v3-base-korean --loss_type bce --learning_rate 1e-5 --batch_size 16 --num_epochs 3 --use_context --max_length 512
    
    log_ok "Phase 1 완료!"
}

# ================================================
# Phase 2: Loss 함수 비교 (RoBERTa 기준)
# ================================================
run_phase2() {
    log_phase "PHASE 2: Loss Function Study"
    
    run_exp "loss-bce" scripts/train.py --model klue/roberta-base --loss_type bce --learning_rate 2e-5 --batch_size 32 --num_epochs 3 --use_context --max_length 512
    run_exp "loss-softbce" scripts/train.py --model klue/roberta-base --loss_type soft_bce --learning_rate 2e-5 --batch_size 32 --num_epochs 3 --use_context --max_length 512
    run_exp "loss-focal" scripts/train.py --model klue/roberta-base --loss_type focal --learning_rate 2e-5 --batch_size 32 --num_epochs 3 --use_context --max_length 512
    run_exp "loss-asl" scripts/train.py --model klue/roberta-base --loss_type asl --learning_rate 2e-5 --batch_size 32 --num_epochs 3 --use_context --max_length 512
    run_exp "loss-weighted" scripts/train.py --model klue/roberta-base --loss_type criterion_weighted --learning_rate 2e-5 --batch_size 32 --num_epochs 3 --use_context --max_length 512
    
    log_ok "Phase 2 완료!"
}

# ================================================
# Phase 3: 고급 아키텍처
# ================================================
run_phase3() {
    log_phase "PHASE 3: Advanced Architecture"
    
    run_exp "arch-multihead" scripts/train_multihead.py --model klue/roberta-base --loss_type soft_bce --learning_rate 2e-5 --batch_size 24 --num_epochs 3 --use_context --max_length 512
    run_exp "arch-crossenc" scripts/train_crossencoder.py --model klue/roberta-base --loss_type soft_bce --learning_rate 2e-5 --batch_size 16 --num_epochs 3 --use_context --max_length 512
    
    log_ok "Phase 3 완료!"
}

# ================================================
# Phase 4: 학습 전략
# ================================================
run_phase4() {
    log_phase "PHASE 4: Learning Strategy"
    
    run_exp "strat-curriculum" scripts/train_curriculum.py --model klue/roberta-base --loss_type soft_bce --learning_rate 2e-5 --batch_size 32 --num_epochs 5 --use_context --max_length 512 --strategy sqrt
    run_exp "strat-contrastive" scripts/train_contrastive.py --model klue/roberta-base --learning_rate 2e-5 --batch_size 32 --num_epochs 3 --use_context --max_length 512 --lambda_contrastive 0.1 --projection_dim 256
    
    log_ok "Phase 4 완료!"
}

# ================================================
# 메인
# ================================================
log_phase "mutsa-seq Experiment (순차 실행)"
log_info "환경: Threadripper PRO 7975WX + RTX 3090"
log_info "GPU 0 단일 사용, 순차 실행"
nvidia-smi --query-gpu=index,memory.used --format=csv

START=$(date +%s)

run_phase1
run_phase2
run_phase3
run_phase4

END=$(date +%s)
MINS=$(( (END - START) / 60 ))

log_phase "COMPLETE"
echo "📊 wandb: https://wandb.ai/dhj9842-hanyang-university/mutsa-seq"
[ -f "$PROGRESS" ] && echo "완료: $(grep -c ':done' "$PROGRESS") 실험"
log_ok "총 소요: ${MINS}분"
