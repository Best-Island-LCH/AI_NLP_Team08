# Curriculum Learning (커리큘럼 학습)

> 쉬운 샘플부터 어려운 샘플까지 단계적으로 학습

---

## 1. 왜 필요한가?

### 문제점: 모든 샘플을 동등하게 학습하면 비효율적

```
우리 데이터의 특성:
  - 만장일치 샘플 (3:0): 명확한 품질 → 쉬움
  - 다수결 샘플 (2:1): 경계선 품질 → 어려움
  
문제:
  1. 어려운 샘플을 초기에 학습하면 모델 혼란
  2. 쉬운 샘플만 학습하면 어려운 케이스 대응 불가
  3. 무작위 학습은 비효율적
```

### 해결책: 인간 학습처럼 쉬운 것부터 어려운 것으로

```
핵심 아이디어:
  "기초부터 심화까지 단계적으로 학습"

학습 순서:
  Phase 1: 만장일치 샘플 (3:0 또는 0:3)
  Phase 2: 강한 다수결 샘플 (3:0 + 2:1 혼합)
  Phase 3: 모든 샘플 포함
```

---

## 2. 어떻게 작동하는가?

### 난이도 측정 방법

```python
# 방법 1: 평가자 일치도 기반
def calculate_agreement_difficulty(evaluations):
    """
    일치도 높음 (3:0) → 쉬움 → 낮은 난이도
    일치도 낮음 (2:1) → 어려움 → 높은 난이도
    """
    yes_count = sum(1 for e in evaluations if e == 'yes')
    no_count = len(evaluations) - yes_count
    
    agreement = max(yes_count, no_count) / len(evaluations)
    difficulty = 1 - agreement  # 일치도 낮으면 어려움
    
    return difficulty

# 방법 2: 모델 손실 기반 (Self-paced Learning)
def calculate_loss_difficulty(model, sample, device):
    """
    손실 높음 → 어려움
    손실 낮음 → 쉬움
    """
    model.eval()
    with torch.no_grad():
        logits = model(sample['input_ids'].to(device), 
                       sample['attention_mask'].to(device))
        loss = F.binary_cross_entropy_with_logits(
            logits, sample['labels'].to(device)
        )
    return loss.item()
```

### 학습 스케줄링

```
커리큘럼 전략들:

1. Fixed Curriculum (고정 커리큘럼)
   - 사전에 난이도 계산
   - 정해진 순서로 학습
   
2. Self-paced Learning (자기 조절 학습)
   - 매 에폭 후 난이도 재계산
   - 현재 잘 맞추는 샘플 위주로 학습
   
3. Baby Steps (아기 걸음)
   - 난이도를 점진적으로 증가
   - Competence 함수로 포함할 샘플 비율 결정
```

---

## 3. 이 태스크에 어떻게 적용하는가?

### 우리 데이터에 맞는 커리큘럼 설계

```
Step 1: 데이터 난이도 분류
  - Easy: 9개 기준 모두 만장일치
  - Medium: 일부 기준이 2:1
  - Hard: 다수 기준이 2:1 또는 기준 간 불일치

Step 2: 학습 스케줄
  - Epoch 1-2: Easy 샘플만
  - Epoch 3-4: Easy + Medium
  - Epoch 5+: 모든 샘플

Step 3: 샘플 가중치
  - Easy: 1.0
  - Medium: epoch에 따라 0→1
  - Hard: epoch에 따라 0→1 (더 느리게)
```

### Competence 함수

```python
def get_competence(epoch, total_epochs, strategy='linear'):
    """
    현재 에폭의 역량(competence) 계산
    역량에 따라 포함할 난이도 결정
    
    Returns:
        0.0 ~ 1.0 값 (높을수록 더 어려운 샘플 포함)
    """
    if strategy == 'linear':
        return min(1.0, (epoch + 1) / total_epochs)
    elif strategy == 'sqrt':
        # 초기에 빠르게, 후기에 느리게
        return min(1.0, np.sqrt((epoch + 1) / total_epochs))
    elif strategy == 'step':
        if epoch < total_epochs // 3:
            return 0.33
        elif epoch < 2 * total_epochs // 3:
            return 0.66
        else:
            return 1.0
```

---

## 4. 예상 효과 및 주의사항

### 예상 효과

```
✅ 학습 초기 안정성 향상
✅ 어려운 샘플에 대한 일반화 향상
✅ 수렴 속도 개선
✅ 최종 성능 향상 (특히 Hard 샘플에서)
```

### 주의사항

```
⚠️ 난이도 정의가 중요 (잘못 정의하면 역효과)
   - 우리 데이터: 평가자 일치도 기반 권장

⚠️ 커리큘럼 속도 조절 필요 (너무 빠르면 효과 없음)
   - 권장: sqrt 또는 step 전략

⚠️ 검증 데이터는 전체 사용해야 함 (커리큘럼 적용 X)

⚠️ Early stopping 주의 (초기에는 validation 성능 낮을 수 있음)
   - 초반 몇 에폭은 건너뛰기
```

---

## 5. 구현 코드

→ [curriculum_learning.py](curriculum_learning.py) 참조

### 빠른 사용법

```python
from curriculum_learning import CurriculumDataset, CurriculumSampler, CurriculumTrainer

# 데이터셋 (난이도 자동 계산)
train_dataset = CurriculumDataset(train_samples, tokenizer)

# 샘플러 (에폭별 샘플 선택)
sampler = CurriculumSampler(
    train_dataset,
    total_epochs=10,
    strategy='sqrt'  # 'linear', 'sqrt', 'step'
)

# 학습기
trainer = CurriculumTrainer(model, train_dataset, val_loader, device, config)
trainer.train()
```

---

## 참고 논문

- [Curriculum Learning](https://dl.acm.org/doi/10.1145/1553374.1553380) (Bengio et al., 2009)
- [Self-Paced Learning](https://papers.nips.cc/paper/2010/hash/e57c6b956a6521b28495f2886ca0977a-Abstract.html) (Kumar et al., 2010)
