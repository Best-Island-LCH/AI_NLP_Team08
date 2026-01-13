# 평가자 불일치 모델링

> 라벨의 불확실성을 학습에 반영하여 더 robust한 모델 학습

---

## 1. 왜 필요한가?

### 문제점: 라벨의 불확실성을 무시

```
현재 방식의 문제:
  
  3명 중 2:1로 평가가 갈린 경우:
    - 다수결: "yes" (2표) → 라벨 = 1
    - 하지만 실제로는 불확실한 샘플!
  
  정보 손실:
    - [3 yes, 0 no] → 1 (확실한 positive)
    - [2 yes, 1 no] → 1 (불확실한 positive)  ← 같은 1로 처리됨!
    - [1 yes, 2 no] → 0 (불확실한 negative)
    - [0 yes, 3 no] → 0 (확실한 negative)
```

### 해결책: 라벨의 불확실성을 학습에 반영

```
방법 1: Soft Labels
  - [3, 0] → 1.0
  - [2, 1] → 0.67
  - [1, 2] → 0.33
  - [0, 3] → 0.0

방법 2: Label Smoothing
  - Hard label을 약간 부드럽게
  - 1 → 0.9, 0 → 0.1

방법 3: 불확실성 모델링
  - 예측값 + 불확실성 동시 출력
  - 불확실한 샘플은 손실 가중치 낮춤
```

---

## 2. 어떻게 작동하는가?

### Soft Label의 직관

```
Hard Label 학습:
  Target: [1, 0]
  → 모델: "이건 확실히 1이야!"
  
Soft Label 학습:
  Target: [0.67, 0.33]
  → 모델: "이건 1일 확률이 높지만, 0일 수도 있어"
  
효과:
  - 모델이 불확실한 샘플에서 극단적 예측을 피함
  - Calibration 향상 (예측 확률이 실제 확률에 가까움)
```

### Label Smoothing의 수식

```
원래 라벨: y ∈ {0, 1}
Smoothed 라벨: y_smooth = y × (1 - α) + α / 2

α = 0.1일 때:
  y = 1 → y_smooth = 0.95
  y = 0 → y_smooth = 0.05
```

### 불확실성 인식 손실

```
아이디어: 불확실한 샘플의 손실 가중치를 낮춤

불확실성 계산:
  soft_target이 0.5에 가까울수록 불확실
  uncertainty = 1 - |soft_target - 0.5| * 2

가중치 적용:
  weighted_loss = loss × (0.5 + 0.5 × certainty)
  
  - 확실한 샘플 (certainty=1): 가중치 1.0
  - 불확실한 샘플 (certainty=0): 가중치 0.5
```

---

## 3. 이 태스크에 어떻게 적용하는가?

### 데이터에서 Soft Label 계산

```python
def compute_soft_labels(row, criteria):
    """
    평가자 투표 비율을 soft label로 변환
    
    예시:
      yes_count=2, no_count=1 → soft_label = 0.67
    """
    soft_labels = []
    
    for c in criteria:
        yes_count = row.get(f'{c}_yes_count', 0)
        no_count = row.get(f'{c}_no_count', 0)
        total = yes_count + no_count
        
        if total > 0:
            soft_label = yes_count / total
        else:
            soft_label = 0.5  # 정보 없으면 불확실
        
        soft_labels.append(soft_label)
    
    return soft_labels
```

### 손실 함수 선택 가이드

```
상황별 추천:

1. 빠른 개선이 필요할 때:
   → Label Smoothing (α=0.1)
   → 구현 간단, 효과 확실

2. 투표 데이터가 있을 때:
   → Soft BCE Loss
   → 정보를 최대한 활용

3. 불확실성 활용이 필요할 때:
   → Uncertainty-Aware Loss
   → 추론 시 불확실성 정보 활용 가능
```

---

## 4. 예상 효과 및 주의사항

### 예상 효과

```
✅ 모델 Calibration 향상 (예측 확률 신뢰도 증가)
✅ 불확실한 샘플에서 과신(overconfidence) 방지
✅ 경계선 케이스 예측 안정화
✅ 테스트 시 불확실성 정보 활용 가능
```

### 주의사항

```
⚠️ Soft label 계산을 위해 원본 투표 데이터 필요
   - 우리 데이터: yes_count, no_count 컬럼 필요

⚠️ Label smoothing 정도(α) 튜닝 필요
   - 시작값: 0.1
   - 범위: 0.05 ~ 0.2

⚠️ 불확실성 모델은 학습이 더 어려움
   - 파라미터 수 증가
   - 수렴 느림

⚠️ 평가 시에는 hard label 기준으로 측정
   - threshold = 0.5 고정
```

---

## 5. 구현 코드

→ [uncertainty_modeling.py](uncertainty_modeling.py) 참조

### 빠른 사용법

```python
from uncertainty_modeling import SoftBCELoss, LabelSmoothingBCELoss, UncertaintyAwareLoss

# 1. Soft BCE Loss (가장 권장)
criterion = SoftBCELoss()
loss = criterion(logits, soft_labels)

# 2. Label Smoothing (간단하게)
criterion = LabelSmoothingBCELoss(smoothing=0.1)
loss = criterion(logits, hard_labels)

# 3. Uncertainty-Aware (고급)
criterion = UncertaintyAwareLoss()
loss = criterion(logits, hard_labels, soft_labels)
```

---

## 참고 논문

- [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567) (Label Smoothing, Szegedy et al., 2016)
- [What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?](https://arxiv.org/abs/1703.04977) (Kendall & Gal, 2017)
