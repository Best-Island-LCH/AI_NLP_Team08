# Multi-task Learning (다중 과제 학습)

> 관련된 태스크들을 함께 학습하여 상호 정보를 활용

---

## 1. 왜 필요한가?

### 문제점: 9개 기준을 독립적으로 학습하면 상관관계를 무시

```
예시: 평가 기준 간 상관관계
  
  consistency (일관성) ←→ sensibleness (적절성)
    "일관된 응답은 대체로 적절하다"
  
  no_hallucination (환각 없음) ←→ specificity (구체성)  
    "환각이 없으면 구체적인 정보를 담고 있다"
  
  linguistic_acceptability ←→ understandability
    "문법적으로 올바르면 이해하기 쉽다"

→ 이런 상관관계를 활용하면 서로 도움!
```

### 해결책: Multi-task Learning으로 상관관계 활용

```
핵심 아이디어:
  "관련된 태스크는 공유된 표현(shared representation)이 도움이 된다"

구조:
  입력 → Shared Encoder → Task-specific Heads
                              ├→ Head1 (linguistic)
                              ├→ Head2 (consistency)
                              └→ ...
```

---

## 2. 어떻게 작동하는가?

### Hard Parameter Sharing vs Soft Parameter Sharing

```
1. Hard Parameter Sharing (우리가 사용):
   
   [입력] → [Shared BERT] → [CLS]
                              │
              ┌───────────────┼───────────────┐
              ↓               ↓               ↓
           [Head1]        [Head2]   ...    [Head9]
              ↓               ↓               ↓
           pred1           pred2           pred9
   
   장점: 파라미터 효율적, 과적합 방지
   단점: 태스크 간 간섭 가능

2. Soft Parameter Sharing:
   
   [입력] → [BERT1] ←정규화→ [BERT2] ←정규화→ [BERT3]
              ↓                 ↓                ↓
           [Head1]           [Head2]          [Head3]
   
   장점: 태스크별 특화 가능
   단점: 파라미터 많음, 학습 복잡
```

### 손실 함수 설계

```
총 손실 = Σ (w_i × L_i)

여기서:
  L_i = i번째 기준의 BCE Loss
  w_i = i번째 기준의 가중치 (중요도 또는 난이도 기반)

가중치 전략:
  1. 균등 가중치: w_i = 1/9 for all i
  2. 난이도 기반: 어려운 기준(no_hallucination)에 높은 가중치
  3. 불확실성 기반: 학습 중 동적으로 가중치 조절 (Kendall et al.)
```

---

## 3. 이 태스크에 어떻게 적용하는가?

### 기준 그룹화 전략

```python
CRITERION_GROUPS = {
    'language': ['linguistic_acceptability', 'understandability'],
    'content': ['consistency', 'sensibleness', 'no_hallucination'],
    'ethics': ['unbias', 'harmlessness'],
    'quality': ['interestingness', 'specificity']
}
```

### 불확실성 기반 가중치 (Kendall et al., 2018)

```
아이디어: 
  - 각 태스크의 "불확실성(σ²)"을 학습 파라미터로
  - 불확실성이 높은 태스크는 가중치가 자동으로 낮아짐

수식:
  L_total = Σ (1/2σ_i² × L_i + log(σ_i))
  
  - 1/2σ² : 정밀도 (불확실성 높으면 가중치 낮음)
  - log(σ) : 정규화 항 (σ가 무한히 커지는 것 방지)
```

---

## 4. 예상 효과 및 주의사항

### 예상 효과

```
✅ 관련 기준 간 정보 공유로 성능 향상
✅ 데이터 효율성 (공유 인코더)
✅ 과적합 방지 (정규화 효과)
✅ 어려운 기준(no_hallucination)도 다른 기준에서 도움 받음
```

### 주의사항

```
⚠️ Negative Transfer: 관련 없는 태스크가 오히려 방해
   → Interaction Layer로 관계 학습 필요

⚠️ 태스크 불균형: 쉬운 태스크가 빨리 수렴하면 어려운 태스크 학습 방해
   → 불확실성 가중치 또는 GradNorm 사용

⚠️ 가중치 튜닝 어려움
   → 자동 가중치 조절 기법 사용 권장
```

---

## 5. 구현 코드

→ [multitask_learning.py](multitask_learning.py) 참조

### 빠른 사용법

```python
from multitask_learning import MultiTaskQualityModel, MultiTaskLoss

# 모델 생성
model = MultiTaskQualityModel(
    model_name='klue/roberta-base',
    num_criteria=9,
    use_interaction=True
)

# 손실 함수 (불확실성 가중치)
criterion = MultiTaskLoss(
    num_tasks=9,
    weighting='uncertainty'  # 'equal', 'fixed', 'uncertainty'
)

# 학습 후 가중치 확인
weights = torch.exp(-criterion.log_vars)
weights = weights / weights.sum()
print("학습된 태스크 가중치:", weights)
```

---

## 참고 논문

- [Multi-Task Learning Using Uncertainty to Weigh Losses](https://arxiv.org/abs/1705.07115) (Kendall et al., 2018)
- [GradNorm: Gradient Normalization for Adaptive Loss Balancing](https://arxiv.org/abs/1711.02257) (Chen et al., 2018)
