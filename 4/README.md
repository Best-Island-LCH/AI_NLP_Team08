# AI 품질 평가 모델 실험

AI 생성 응답의 품질을 9가지 기준으로 평가하는 Multi-Label Classification 실험 프로젝트입니다.

## 1. Overview

한국어 BERT 계열 모델을 활용하여 AI 응답의 품질을 평가합니다.

---

## 2. EDA (탐색적 데이터 분석)

> `eda/` 폴더에 데이터셋 분석 결과가 정리되어 있습니다.

### 데이터셋 규모

| 항목 | Train | Validation |
|------|-------|------------|
| 샘플 수 | 400,572 | 50,047 |
| 평가 기준 | 9개 | 9개 |
| 평가자 수 | 3명 | 3명 |

### 핵심 발견

| 분석 항목 | 주요 발견 | 시사점 |
|-----------|----------|--------|
| **클래스 불균형** | Positive 79.6%~92.2%로 압도적 | Focal Loss / ASL 적용 필요 |
| **평가자 일치도** | `no_hallucination` 불일치 35.5%로 가장 높음 | Soft Labels 사용 권장 |
| **기준 간 상관관계** | interestingness ↔ specificity (0.909) | Multi-Head 그룹화 가능 |
| **멀티턴 영향** | `no_hallucination` -15.7% 품질 하락 | 대화 맥락 포함 필수 |
| **샘플 난이도** | 90.4%가 쉬운 샘플 | Curriculum Learning 효과적 |

### 기준 그룹화

```
Group A (Content Quality): interestingness, specificity     → Focal/ASL Loss
Group B (Safety): unbias, harmlessness                      → BCE Loss  
Group C (Coherence): consistency, no_hallucination          → Soft BCE + 높은 가중치
Group D (Independent): linguistic_acceptability, understandability, sensibleness
```

> 자세한 분석 결과는 [`eda/eda_summary.md`](eda/eda_summary.md) 참고

---

### 평가 기준 (9개)

| 기준 | 설명 |
|------|------|
| linguistic_acceptability | 문법적 정확성 |
| consistency | 일관성 |
| interestingness | 흥미도 |
| unbias | 편향 없음 |
| harmlessness | 무해성 |
| no_hallucination | 환각 없음 |
| understandability | 이해 가능성 |
| sensibleness | 합리성 |
| specificity | 구체성 |

## 3. Installation

```bash
# 가상환경 생성 (권장)
conda create -n mutsa python=3.10
conda activate mutsa

# 의존성 설치
pip install -r requirements.txt
```

## 4. Project Structure

```
.
├── eda/                        # 탐색적 데이터 분석
│   ├── eda_summary.md         # EDA 분석 결과 정리
│   └── figures/               # 분석 결과 시각화 (...)
│
└── experiments/                # 실험 코드
    ├── config/                 # 설정 파일
    │   ├── config.yaml        # 기본 학습 설정
    │   ├── experiments.yaml   # 실험 Phase 정의
    │   ├── model_configs.yaml # 모델별 하이퍼파라미터
    │   └── sweep_config.yaml  # W&B Sweep 설정
    ├── scripts/                # 실행 스크립트
    │   ├── train.py           # 기본 학습
    │   ├── train_multihead.py # Multi-Head 아키텍처
    │   ├── train_crossencoder.py  # Cross-Encoder
    │   ├── train_curriculum.py    # Curriculum Learning
    │   ├── train_contrastive.py   # Contrastive Learning
    │   ├── run_parallel.py    # 2-GPU 병렬 실험 오케스트레이터
    │   ├── evaluate.py        # 평가 스크립트
    │   └── visualize_results.py   # 결과 시각화
    ├── src/                    # 소스 코드
    │   ├── data/              # 데이터 처리
    │   ├── models/            # 모델 아키텍처
    │   ├── training/          # 학습 로직 & 손실 함수
    │   ├── evaluation/        # 평가 메트릭
    │   └── utils/             # 유틸리티
    └── requirements.txt        # 의존성
```

## 5. Experiment Phases

### Phase 1: 아키텍처 백본 비교 (10 실험)
5가지 모델 × 맥락 유/무 비교

| 모델 | ID | 파라미터 | 특징 |
|------|-----|----------|------|
| BERT | klue/bert-base | 111M | Original bidirectional encoder |
| RoBERTa | klue/roberta-base | 111M | Dynamic masking, no NSP |
| ELECTRA | monologg/koelectra-base-v3-discriminator | 110M | Discriminator-based pretraining |
| DistilBERT | monologg/distilkobert | 28M | Lightweight, fast inference |
| DeBERTa | team-lucid/deberta-v3-base-korean | 86M | Disentangled attention, SOTA |

### Phase 2: 손실 함수 비교 (10 실험)
Phase 1 Top-2 모델 × 5가지 Loss

- **BCE**: Binary Cross Entropy
- **Soft BCE**: Vote ratio 기반 Soft Label
- **Focal Loss**: Class imbalance 대응
- **ASL**: Asymmetric Loss (극심한 불균형)
- **Criterion Weighted**: EDA 기반 기준별 가중치

### Phase 3: 아키텍처 (3 실험)
- Multi-Head Classification
- Cross-Encoder
- Standard (비교 대조군)

### Phase 4: 학습 전략 (3 실험)
- Curriculum Learning (sqrt strategy)
- Contrastive Learning
- Combined approach

## 6. Usage

### 단일 실험 실행

```bash
cd experiments

# 기본 학습
python scripts/train.py \
    --model_name klue/roberta-base \
    --loss_type bce \
    --batch_size 32 \
    --run_name my-experiment

# 맥락 포함 학습
python scripts/train.py \
    --model_name klue/roberta-base \
    --use_context \
    --max_length 512 \
    --batch_size 16 \
    --run_name roberta-with-context
```

### 전체 실험 (2-GPU 병렬)

```bash
# nohup으로 백그라운드 실행 (SSH 종료 후에도 계속)
cd experiments
nohup python scripts/run_parallel.py > logs/parallel.log 2>&1 &
disown
```

### GPU 지정 실행

```bash
# 특정 GPU 사용
python scripts/train.py --gpu 1 --model_name klue/bert-base
```

## 7. Weights & Biases

실험 추적은 [W&B](https://wandb.ai/)를 통해 진행됩니다.

```bash
# W&B 로그인 (최초 1회)
wandb login
```

`config/config.yaml`에서 설정:

```yaml
wandb:
  entity: "your-entity"
  project: "mutsa-v2"
  enabled: true
```

## 8. Configuration

### 주요 설정 (`config/config.yaml`)

| 항목 | 기본값 | 설명 |
|------|--------|------|
| `model.name` | klue/roberta-base | 사전학습 모델 |
| `tokenizer.max_length` | 128 | 최대 토큰 길이 |
| `training.batch_size` | 32 | 배치 크기 |
| `training.learning_rate` | 2e-5 | 학습률 |
| `training.num_epochs` | 5 | 에폭 수 |
| `training.precision` | bf16 | Mixed Precision |
| `loss.type` | soft_bce | 손실 함수 |

### 대화 맥락 설정

```yaml
context:
  enabled: true
  max_prev_turns: 7
  max_length: 512
```

## 9. Metrics

- **Macro F1**: 주요 평가 지표
- **Per-criterion F1**: 9개 기준별 F1 점수
- **AUC-ROC**: 각 기준별 ROC 곡선
- **Threshold Optimization**: 기준별 최적 임계값 탐색

## 10. Advanced Features

### Early Stopping
```yaml
early_stopping:
  enabled: true
  patience: 3
  metric: "eval_macro_f1"
  mode: "max"
```

### Gradient Clipping
```yaml
training:
  max_grad_norm: 1.0
```

### Learning Rate Scheduler
```yaml
training:
  lr_scheduler_type: "cosine"  # linear, cosine, polynomial
```
