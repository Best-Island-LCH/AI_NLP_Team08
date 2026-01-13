# Multi-Head Classification (기준별 독립 분류)

> 각 평가 기준에 특화된 분류기 + 기준 간 상호작용

---

## 1. 기존 방식의 한계

```
베이스라인:
  BERT → [CLS] → Linear(hidden, 9) → 9개 예측
  
문제점:
  1. 하나의 Linear 레이어로 9개 기준 동시 예측
  2. 기준별 특성 무시
     - linguistic_acceptability: 문법적 특성 중요
     - no_hallucination: 사실 관계 중요
     - interestingness: 내용적 특성 중요
  3. 기준 간 관계 활용 부족
```

---

## 2. 새로운 아키텍처의 아이디어

```
핵심 아이디어: 기준별 전문화 + 상호작용

  Step 1: 기준별 독립 헤드
    각 기준에 특화된 분류기 학습
  
  Step 2: 기준 그룹화
    관련 기준끼리 그룹화하여 정보 공유
    - 언어적 그룹: linguistic, understandability
    - 내용적 그룹: consistency, sensibleness, no_hallucination
    - 윤리적 그룹: unbias, harmlessness
    - 품질 그룹: interestingness, specificity
  
  Step 3: 기준 간 상호작용
    그룹 내/간 정보 교환
```

---

## 3. 구조 다이어그램

```
┌─────────────────────────────────────────────────────────────────────┐
│                  Multi-Head Classification Model                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│                         Input Text                                   │
│                             │                                        │
│                             ▼                                        │
│                      ┌───────────┐                                   │
│                      │   BERT    │                                   │
│                      │ Encoder   │                                   │
│                      └─────┬─────┘                                   │
│                            │                                         │
│                         [CLS]                                        │
│                            │                                         │
│     ┌──────────────────────┼──────────────────────┐                 │
│     │                      │                      │                  │
│     ▼                      ▼                      ▼                  │
│ ┌─────────┐          ┌─────────┐          ┌─────────┐               │
│ │Language │          │Content  │          │Ethics   │ ...           │
│ │ Group   │          │ Group   │          │ Group   │               │
│ └────┬────┘          └────┬────┘          └────┬────┘               │
│      │                    │                    │                     │
│ ┌────┴────┐          ┌────┴────┐          ┌────┴────┐               │
│ │Head1    │          │Head3    │          │Head5    │               │
│ │Head2    │          │Head4    │          │Head6    │               │
│ └────┬────┘          └────┬────┘          └────┬────┘               │
│      │                    │                    │                     │
│      ▼                    ▼                    ▼                     │
│   [pred1,2]           [pred3,4]           [pred5,6]    ...          │
│                                                                      │
│     └────────────────────┬─────────────────────┘                    │
│                          │                                           │
│                          ▼                                           │
│                ┌─────────────────┐                                   │
│                │   Interaction   │  (선택적)                         │
│                │     Layer       │                                   │
│                └────────┬────────┘                                   │
│                         │                                            │
│                         ▼                                            │
│               [Final 9 predictions]                                  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. 기준 그룹화

```python
CRITERION_GROUPS = {
    'language': [
        'linguistic_acceptability',  # 문법적 올바름
        'understandability'          # 이해 용이성
    ],
    'content': [
        'consistency',      # 일관성
        'sensibleness',     # 적절성
        'no_hallucination'  # 사실 정확성
    ],
    'ethics': [
        'unbias',      # 편향 없음
        'harmlessness' # 유해성 없음
    ],
    'quality': [
        'interestingness',  # 흥미로움
        'specificity'       # 구체성
    ]
}
```

---

## 5. 장단점 및 적용 시나리오

### 장점

```
✅ 기준별 특화 학습 가능
✅ 그룹 내 정보 공유로 효율성
✅ 기준별 성능 분석 용이
✅ 특정 기준만 fine-tuning 가능
```

### 단점

```
❌ 파라미터 수 증가
❌ 그룹 정의가 주관적일 수 있음
❌ 기준 간 과도한 분리는 정보 손실
```

### 적용 시나리오

```
- 기준별 성능 차이가 큰 경우
- 특정 기준 성능 향상이 목표인 경우
- 해석 가능성이 중요한 경우
```

---

## 6. Focal Loss

어려운 샘플(잘 맞추지 못하는)에 더 큰 가중치를 부여:

```
L = -α(1-p)^γ * log(p)  (positive)
L = -(1-α)p^γ * log(1-p)  (negative)

파라미터:
  - α: 클래스 불균형 조절 (기본 0.25)
  - γ: 어려운 샘플 강조 정도 (기본 2.0)
```

---

## 7. 구현 코드

→ [multihead_classification.py](multihead_classification.py) 참조

### 빠른 사용법

```python
from multihead_classification import MultiHeadClassificationModel, FocalLoss

# 모델 생성
model = MultiHeadClassificationModel(
    model_name='klue/roberta-base',
    use_group_interaction=True
)

# Focal Loss
criterion = FocalLoss(alpha=0.25, gamma=2.0)

# 학습
logits = model(input_ids, attention_mask)
loss = criterion(logits, labels)
```

---

## 참고 논문

- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) (Lin et al., 2017)
- [Multi-Task Learning](https://arxiv.org/abs/1706.05098) (Ruder, 2017)
