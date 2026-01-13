# 사전학습 모델 비교 가이드

> 한국어 NLP 태스크를 위한 BERT 계열 모델 비교 및 선택 가이드

---

## 1. 모델 비교표

| 모델 | HuggingFace ID | 파라미터 | 특징 | 추천 상황 |
|------|----------------|----------|------|-----------|
| **KLUE-RoBERTa** | `klue/roberta-base` | 111M | 한국어 위키+뉴스 학습, 안정적 | 베이스라인, 범용 |
| **KLUE-RoBERTa-large** | `klue/roberta-large` | 337M | 더 큰 용량, 높은 성능 | 성능 우선 |
| **KoELECTRA** | `monologg/koelectra-base-v3-discriminator` | 111M | 효율적 사전학습, 빠른 수렴 | 빠른 실험 |
| **DeBERTa-v3** | `microsoft/deberta-v3-base` | 184M | Disentangled Attention, SOTA | 최고 성능 |
| **XLM-RoBERTa** | `xlm-roberta-base` | 278M | 100+ 언어 지원 | 다국어 확장 |

---

## 2. 모델별 상세 설명

### KLUE-RoBERTa (추천: 베이스라인) ⭐

```
장점:
  ✅ 한국어에 최적화된 토크나이저
  ✅ KLUE 벤치마크에서 검증됨
  ✅ 풍부한 한국어 사전학습 데이터
  ✅ base/large 둘 다 제공

단점:
  ❌ 영어 혼합 텍스트에 약함
  ❌ base 모델은 복잡한 추론에 한계

권장 하이퍼파라미터:
  - Learning Rate: 2e-5 ~ 3e-5
  - Batch Size: 16 ~ 32
  - Warmup Ratio: 0.1
```

### DeBERTa-v3 (추천: 성능 우선) ⭐⭐

```
장점:
  ✅ Disentangled Attention: 위치와 내용 정보 분리
  ✅ Enhanced Mask Decoder: 더 나은 MLM 학습
  ✅ 대부분의 NLU 태스크에서 SOTA
  ✅ 다국어 버전으로 한국어도 가능

단점:
  ❌ 추론 속도가 상대적으로 느림
  ❌ 한국어 전용 버전 없음 (다국어 사용)
  ❌ 메모리 사용량 더 많음

권장 하이퍼파라미터:
  - Learning Rate: 1e-5 ~ 2e-5
  - Batch Size: 8 ~ 16
  - Warmup Ratio: 0.1
```

### KoELECTRA (추천: 효율성 우선)

```
장점:
  ✅ ELECTRA 방식: 작은 Generator + Discriminator
  ✅ 같은 크기 대비 빠른 학습
  ✅ 한국어 특화
  ✅ 수렴 속도 빠름

단점:
  ❌ 대형 모델 버전 없음
  ❌ 일부 태스크에서 RoBERTa보다 낮은 성능
  ❌ 생성(Generation) 태스크에 부적합

권장 하이퍼파라미터:
  - Learning Rate: 3e-5 ~ 5e-5
  - Batch Size: 16 ~ 32
  - Warmup Ratio: 0.05
```

### XLM-RoBERTa (추천: 다국어)

```
장점:
  ✅ 100+ 언어 지원
  ✅ Zero-shot 다국어 전이 가능
  ✅ 대규모 다국어 코퍼스 학습

단점:
  ❌ 단일 언어 모델보다 성능 낮을 수 있음
  ❌ 모델 크기가 큼
  ❌ 한국어 특화 최적화 없음

권장 하이퍼파라미터:
  - Learning Rate: 2e-5 ~ 3e-5
  - Batch Size: 16 ~ 32
  - Warmup Ratio: 0.1
```

---

## 3. 모델 선택 플로우차트

```
┌─────────────────────────────────────────────────────────────┐
│                    모델 선택 플로우차트                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Q1. GPU 메모리가 충분한가? (16GB+)                          │
│      │                                                      │
│      ├─ Yes → Q2. 최고 성능이 필요한가?                      │
│      │         ├─ Yes → DeBERTa-v3-base                     │
│      │         └─ No  → KLUE-RoBERTa-large                  │
│      │                                                      │
│      └─ No  → Q3. 빠른 실험이 필요한가?                      │
│                ├─ Yes → KoELECTRA-base                      │
│                └─ No  → KLUE-RoBERTa-base                   │
│                                                             │
│  특수 케이스:                                                │
│  - 다국어 필요 → XLM-RoBERTa                                 │
│  - 긴 시퀀스 (>512) → Longformer 또는 BigBird               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. 모델 로드 코드

```python
from transformers import AutoModel, AutoTokenizer

# 모델 선택 매핑
MODEL_CONFIGS = {
    'klue-roberta': 'klue/roberta-base',
    'klue-roberta-large': 'klue/roberta-large',
    'koelectra': 'monologg/koelectra-base-v3-discriminator',
    'deberta': 'microsoft/deberta-v3-base',
    'xlm-roberta': 'xlm-roberta-base',
}

def load_model(model_key='klue-roberta'):
    """
    모델과 토크나이저 로드
    
    Args:
        model_key: 모델 키 (위 MODEL_CONFIGS 참조)
    
    Returns:
        model: 사전학습 모델
        tokenizer: 토크나이저
    """
    model_name = MODEL_CONFIGS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    print(f"✅ Loaded: {model_name}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, tokenizer


# 사용 예시
if __name__ == "__main__":
    # 기본 모델 로드
    model, tokenizer = load_model('klue-roberta')
    
    # 테스트 토크나이징
    text = "AI 응답의 품질을 평가합니다."
    tokens = tokenizer(text, return_tensors='pt')
    print(f"   토큰 수: {tokens['input_ids'].shape[1]}")
```

---

## 5. 성능 벤치마크 (참고)

### KLUE 벤치마크 결과

| 모델 | NLI | STS | RE | NER | 평균 |
|------|-----|-----|-----|-----|------|
| KLUE-RoBERTa-base | 81.5 | 89.2 | 66.1 | 85.3 | 80.5 |
| KLUE-RoBERTa-large | 85.2 | 91.1 | 69.8 | 87.1 | 83.3 |
| KoELECTRA-base | 80.8 | 88.5 | 65.2 | 84.6 | 79.8 |
| XLM-RoBERTa-base | 79.2 | 87.1 | 62.4 | 82.9 | 77.9 |

> 출처: KLUE Paper, 우리 태스크와 직접적인 관련은 참고용

---

## 6. 우리 태스크를 위한 권장사항

```
1순위: KLUE-RoBERTa-base
  - 안정적이고 검증됨
  - 한국어 대화에 적합
  - 리소스 효율적

2순위: DeBERTa-v3-base
  - 최고 성능이 필요할 때
  - GPU 메모리 16GB+ 필요
  - 추론 속도 10~20% 느림

3순위: KLUE-RoBERTa-large
  - 성능과 효율의 균형
  - GPU 메모리 12GB+ 필요

실험 순서:
  1. KLUE-RoBERTa-base로 베이스라인 구축
  2. 성능 부족 시 DeBERTa 시도
  3. 속도가 중요하면 KoELECTRA 시도
```

---

## 참고 링크

- [KLUE GitHub](https://github.com/KLUE-benchmark/KLUE)
- [HuggingFace Korean Models](https://huggingface.co/models?language=ko)
- [DeBERTa Paper](https://arxiv.org/abs/2006.03654)
- [ELECTRA Paper](https://arxiv.org/abs/2003.10555)
