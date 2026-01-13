# Contrastive Learning (대조 학습)

> 좋은 응답끼리는 가깝게, 나쁜 응답과는 멀게 임베딩 공간을 학습

---

## 1. 왜 필요한가?

### 문제점: 일반적인 분류 학습의 한계

```
일반 분류 학습:
  입력 → BERT → [CLS] → Linear → Sigmoid → 0 or 1

문제:
  1. 품질의 "정도"를 구분하지 못함
     - 아주 좋은 응답 vs 약간 좋은 응답 = 둘 다 1
  2. 경계선 샘플(2:1로 갈린 평가)에서 불안정
  3. 임베딩 공간에서 품질별 구분이 명확하지 않음
```

### 해결책: Contrastive Learning

```
핵심 아이디어:
  "좋은 응답끼리는 가깝게, 나쁜 응답과는 멀게" 임베딩

효과:
  1. 임베딩 공간에서 품질별 클러스터 형성
  2. 경계선 샘플도 연속적인 공간에서 자연스럽게 배치
  3. 더 robust한 표현 학습
```

---

## 2. 어떻게 작동하는가?

### InfoNCE Loss

```
수식:
  L = -log[ exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ) ]

직관적 설명:
  - z_i: 현재 샘플의 임베딩
  - z_j: 같은 품질(positive)의 임베딩
  - z_k: 다른 품질(negative)의 임베딩
  - τ: temperature (보통 0.07~0.1)
  
  → z_i와 z_j는 가깝게 (분자 ↑)
  → z_i와 z_k는 멀게 (분모에서 작게)
```

### 시각적 이해

```
학습 전 임베딩 공간:          학습 후 임베딩 공간:
                              
    ●  ○  ●                      ●●●
  ○    ●    ○                   ●●●●
    ●  ○  ●                     
                                 ○○○
  ● = 좋은 응답                   ○○○○
  ○ = 나쁜 응답                  
                              (클러스터 형성!)
```

---

## 3. 이 태스크에 어떻게 적용하는가?

### Positive/Negative Pair 구성 전략

```python
# 전략 1: 같은 기준 내에서 품질별 대조
# - Positive: 같은 품질 (둘 다 good 또는 둘 다 bad)
# - Negative: 다른 품질 (good vs bad)

# 전략 2: 만장일치 샘플 기반
# - Positive: 3명 모두 yes인 샘플끼리
# - Hard Negative: 2:1로 갈린 샘플
# - Easy Negative: 3명 모두 no인 샘플
```

### 모델 구조

```
입력 → BERT → [CLS]
                 │
    ┌────────────┴────────────┐
    ↓                         ↓
Projection Head           Classification Head
    ↓                         ↓
Contrastive Loss          BCE Loss

Total Loss = BCE Loss + λ × Contrastive Loss
```

---

## 4. 예상 효과 및 주의사항

### 예상 효과

```
✅ 임베딩 품질 향상 → 더 나은 분류 성능
✅ 경계선 샘플에서 안정적인 예측
✅ 새로운 도메인으로의 일반화 향상
```

### 주의사항

```
⚠️ 배치 크기가 커야 효과적 (negative 샘플 다양성)
   - 권장: batch_size >= 32
   
⚠️ Temperature 하이퍼파라미터 튜닝 필요
   - 시작값: 0.07
   - 범위: 0.05 ~ 0.2
   
⚠️ 학습 초기에는 불안정할 수 있음
   - Warm-up 필요
   
⚠️ lambda_contrastive 가중치 조절 필요
   - 시작값: 0.1
   - 범위: 0.05 ~ 0.5
```

---

## 5. 구현 코드

→ [contrastive_learning.py](contrastive_learning.py) 참조

### 빠른 사용법

```python
from contrastive_learning import ContrastiveQualityModel, ContrastiveTrainer

# 모델 생성
model = ContrastiveQualityModel(
    model_name='klue/roberta-base',
    num_labels=9,
    projection_dim=256,
    temperature=0.07
)

# 학습 설정
config = {
    'learning_rate': 2e-5,
    'lambda_contrastive': 0.1,
    'batch_size': 32,
    'epochs': 5
}

# 학습
trainer = ContrastiveTrainer(model, train_loader, val_loader, device, config)
results = trainer.train_epoch()
```

---

## 참고 논문

- [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362) (Khosla et al., 2020)
- [SimCLR](https://arxiv.org/abs/2002.05709) (Chen et al., 2020)
