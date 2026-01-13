# Cross-Encoder (컨텍스트-응답 상호작용)

> Context와 Response의 상호작용을 명시적으로 모델링

---

## 1. 기존 방식의 한계

```
베이스라인:
  "[Context] [SEP] [Response]" → BERT → [CLS] → 분류
  
문제점:
  1. Context와 Response가 단순 연결
  2. 질문의 "어떤 부분"에 응답이 대응하는지 불명확
  3. 상호작용 정보 부족
  
예시:
  Q: "서울의 오늘 날씨와 내일 날씨 알려줘"
  A: "오늘은 맑고, 내일은 비가 올 예정입니다."
  
  → "오늘 날씨" ↔ "오늘은 맑고" 대응 관계를 명시적으로 모델링하면 좋음
```

---

## 2. 새로운 아키텍처의 아이디어

```
핵심 아이디어: Cross-Attention으로 상호작용 모델링

  Step 1: Context와 Response를 각각 인코딩
    Context → BERT → context_hidden  [seq_len_c, hidden]
    Response → BERT → response_hidden  [seq_len_r, hidden]
  
  Step 2: Cross-Attention
    Query: Response
    Key, Value: Context
    
    → Response의 각 토큰이 Context의 어떤 부분을 참조하는지 학습
  
  Step 3: 결합 및 분류
    cross_attended + response → Classifier
```

---

## 3. 구조 다이어그램

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Cross-Encoder Architecture                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│        Context                              Response                 │
│   "서울의 오늘 날씨와                    "오늘은 맑고,              │
│    내일 날씨 알려줘"                      내일은 비가..."            │
│            │                                    │                    │
│            ▼                                    ▼                    │
│     ┌───────────┐                        ┌───────────┐              │
│     │   BERT    │                        │   BERT    │              │
│     │ (shared)  │                        │ (shared)  │              │
│     └─────┬─────┘                        └─────┬─────┘              │
│           │                                    │                    │
│           ▼                                    ▼                    │
│      Context_H                            Response_H                │
│    [seq_c, hidden]                       [seq_r, hidden]            │
│           │                                    │                    │
│           │         ┌─────────────┐            │                    │
│           └────────►│   Cross     │◄───────────┘                    │
│                     │  Attention  │                                  │
│                     │  (R → C)    │                                  │
│                     └──────┬──────┘                                  │
│                            │                                         │
│                            ▼                                         │
│                    Cross-Attended                                    │
│                    Response Repr                                     │
│                            │                                         │
│                    ┌───────┴───────┐                                │
│                    │   Combine &   │                                │
│                    │   Classify    │                                │
│                    └───────────────┘                                │
│                            │                                         │
│                            ▼                                         │
│                    [9 Quality Scores]                                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. 장단점 및 적용 시나리오

### 장점

```
✅ Context-Response 상호작용 명시적 모델링
✅ 질문의 어떤 부분에 응답이 대응하는지 학습
✅ 어텐션 시각화로 해석 가능
✅ no_hallucination 등 대응 관계가 중요한 기준에 효과적
```

### 단점

```
❌ 두 번 인코딩으로 계산량 증가
❌ 구현 복잡도 증가
❌ 짧은 Context에서는 이점 적음
```

### 적용 시나리오

```
- 질문-응답 대응 관계가 중요한 경우
- 사실 검증(no_hallucination)이 중요한 경우
- 상세한 해석이 필요한 경우
```

---

## 5. 구현 코드

→ [cross_encoder.py](cross_encoder.py) 참조

### 빠른 사용법

```python
from cross_encoder import CrossEncoderModel, CrossEncoderDataset

# 모델 생성
model = CrossEncoderModel(
    model_name='klue/roberta-base',
    num_labels=9,
    num_cross_layers=2
)

# 추론
context = "서울의 오늘 날씨와 내일 날씨 알려줘"
response = "오늘 서울은 맑고 기온 20도입니다."

context_enc = tokenizer(context, return_tensors='pt', ...)
response_enc = tokenizer(response, return_tensors='pt', ...)

logits, attn_weights = model(
    context_enc['input_ids'], context_enc['attention_mask'],
    response_enc['input_ids'], response_enc['attention_mask']
)
```

---

## 참고 논문

- [Cross-Encoders for Sentence Pair Scoring](https://arxiv.org/abs/1908.10084)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Transformer)
