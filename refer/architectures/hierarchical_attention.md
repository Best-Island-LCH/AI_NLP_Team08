# Hierarchical Attention (계층적 어텐션)

> 대화 구조를 명시적으로 모델링하여 중요한 턴에 집중

---

## 1. 기존 방식의 한계

```
베이스라인 구조:
  [Turn1] [Turn2] [Turn3] ... [턴들 연결] → BERT → [CLS] → 분류

문제점:
  1. 모든 토큰을 동등하게 처리
     - 중요한 턴과 덜 중요한 턴 구분 없음
  2. 긴 대화에서 정보 손실
     - BERT 최대 512 토큰 제한
     - 앞부분 컨텍스트 잘림
  3. 대화 구조 무시
     - 질문-답변 쌍의 구조적 정보 활용 안 함
```

---

## 2. 새로운 아키텍처의 아이디어

```
핵심 아이디어: 2단계 어텐션

  Step 1: 각 턴을 독립적으로 인코딩
    Turn1 → BERT → turn1_repr
    Turn2 → BERT → turn2_repr
    ...
  
  Step 2: 턴 레벨 어텐션으로 중요한 턴 찾기
    [turn1_repr, turn2_repr, ...] → Turn Attention → important_repr
  
  Step 3: 중요한 턴 내에서 토큰 레벨 어텐션
    important_repr → Token Attention → final_repr

효과:
  - 중요한 턴에 더 집중
  - 대화 구조 반영
  - 긴 대화도 효과적으로 처리
```

---

## 3. 구조 다이어그램

```
입력: [Turn1: "질문1"] [Turn2: "응답1"] [Turn3: "질문2"] [Turn4: "응답2(평가대상)"]

┌─────────────────────────────────────────────────────────────────────┐
│                    Hierarchical Attention Model                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Turn1    Turn2    Turn3    Turn4                                   │
│    │        │        │        │                                      │
│    ▼        ▼        ▼        ▼                                      │
│ ┌──────┐┌──────┐┌──────┐┌──────┐                                    │
│ │ BERT ││ BERT ││ BERT ││ BERT │  (공유 BERT)                       │
│ └──┬───┘└──┬───┘└──┬───┘└──┬───┘                                    │
│    │        │        │        │                                      │
│    ▼        ▼        ▼        ▼                                      │
│  [CLS1]  [CLS2]  [CLS3]  [CLS4]   ← 각 턴의 대표 벡터               │
│    │        │        │        │                                      │
│    └────────┴────┬───┴────────┘                                      │
│                  │                                                    │
│                  ▼                                                    │
│         ┌──────────────┐                                             │
│         │ Turn-level   │                                             │
│         │ Attention    │  ← "어떤 턴이 중요한가?"                    │
│         └──────┬───────┘                                             │
│                │                                                      │
│                ▼                                                      │
│         Weighted Turn                                                 │
│         Representation                                                │
│                │                                                      │
│                ▼                                                      │
│         ┌──────────────┐                                             │
│         │ Classifier   │                                             │
│         │ (9 outputs)  │                                             │
│         └──────────────┘                                             │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. 장단점 및 적용 시나리오

### 장점

```
✅ 대화 구조를 명시적으로 모델링
✅ 긴 대화도 효과적으로 처리
✅ 어텐션 가중치로 해석 가능 (어떤 턴이 중요했는지)
✅ 중요한 컨텍스트에 집중
```

### 단점

```
❌ 구현 복잡도 증가
❌ 학습 시간 증가 (여러 번 인코딩)
❌ 메모리 사용량 증가
```

### 적용 시나리오

```
- 대화가 길고 (5턴 이상) 컨텍스트가 중요한 경우
- 어텐션 가중치를 통한 설명이 필요한 경우
- 베이스라인 대비 성능 향상이 필요한 경우
```

---

## 5. 구현 코드

→ [hierarchical_attention.py](hierarchical_attention.py) 참조

### 빠른 사용법

```python
from hierarchical_attention import HierarchicalAttentionModel, HierarchicalDataProcessor

# 모델 생성
model = HierarchicalAttentionModel(
    model_name='klue/roberta-base',
    num_labels=9,
    max_turns=8
)

# 데이터 처리기
processor = HierarchicalDataProcessor(
    tokenizer=tokenizer,
    max_turn_length=128,
    max_turns=8
)

# 대화 처리
utterances = [
    {'speaker': 'human', 'text': '오늘 날씨 어때?'},
    {'speaker': 'bot', 'text': '서울은 맑고 20도입니다.'}
]
turn_ids, turn_masks, valid_mask = processor.process_conversation(utterances)

# 추론
logits, attn_weights = model(turn_ids, turn_masks, valid_mask)
```

---

## 참고 논문

- [Hierarchical Attention Networks for Document Classification](https://www.aclweb.org/anthology/N16-1174/) (Yang et al., 2016)
- [BERT for Dialogue](https://arxiv.org/abs/1907.11692) (Mehri et al., 2019)
