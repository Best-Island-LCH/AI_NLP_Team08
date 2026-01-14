# Decision Process
## 1. 왜 BERT 계열 모델인가?
AI 품질 평가(분류)와 같은 NLU(자연어 이해)작업에서는,
BERT 계열(Encoder-only)모델이 가성비와 성능 면에서 가장 유리한 선택이다.
특정 목적을 가진 평가기(Evaluator)모델을 직접 학습시킬 때는 여전히 BERT 계열이 주인공이다.

### 1. 왜 평가에 더 유리할까?
""양방향성(Bidirectional)""이라는 강력한 무기를 가지고 있다.  

- ``BERT``: 문장 전체를 한꺼번에 읽는다. "이 답변이 질문에 맞는가?"를 판단할 때 문맥의 <strong>앞뒤를 동시에 보는 능력.</strong>
=> 우리는 "멀티태스크 학습"을 효율적으로 수행해야 한다. RoBERTa는 BERT의 학습 방식을 개선하여 데이터 효율성이 높고, 베이스 모델 기준 약 1.1억 개의 파라미터로 구성되어 있어 T4 GPU 환경에서도 빠른 실험 루프와 추론 속도를 보장한다.

- ``GPT``: 문장을 왼쪽에서 오른쪽으로 한 단어씩 읽으며 다음 단어를 '생성'하는데 특화되어 있다. 
구조적으로 Masking이 있어 이전 단어만 볼 수 있는 제약으로 문맥 이해의 균형감이 인코더보다 떨어질 수 있다.
=> 학습 속도가 너무 느려 단기간 프로젝트에 적합X
- ``T5/BART(Encoder-Decoder)``: 번역이나 요약처럼 입력을 받아 새로운 문장을 만들 때는 최강이지만, 단순히 0과 1을 맞히는 분류 작업에는 엔진이 너무 무겁고 복잡하다.
=> 텍스트를 텍스트로 내뱉기 때문에, 0과 1이라는 수치로 정확하게 성능을 측정하기 번거로움.


# 평가 1
## 사용한 모델: klue/roberta-base

### Cross-Encoder 형식, Multi-label 방식
```
full_input = f"{history_text}{current_row['human_question']} [SEP] {current_row['bot_response']}"
```
두 개 이상의 문장을 하나의 입력창에 동시에 넣어 '크로스-인코더'
-> 문장 간의 관계를 Cross해서 읽을 수 있도록 데이터 포맷.

```
target_cols = [f'{c}_majority' for c in CRITERIA]
```
학습 데이터의 정답(Label)이 단일 숫자가 아니라 0과 1로 구성된 9차원의 벡터이다.
```
# 추론 코드에서
probabilities = torch.sigmoid(logits) 
predictions = (probabilities > 0.5).astype(int)
```
각 라벨의 확률을 0에서 1사이로 독립적으로 계산. '멀티-레이블'
### 더 생각해 볼 수 있는 변화? => DeBERTa-v3 모델 사용
굳이 어려운 ``Hierarchical``로 넘어가기보다, ``DeBERTa-v3``모델을 사용하여 Cross-Encoder의 효율을 극대화.

```
# Validation 데이터로 평가
eval_results = trainer.evaluate()

print("=" * 50)
print("평가 결과")
print("=" * 50)
for key, value in eval_results.items():
    if 'loss' in key or 'f1' in key or 'match' in key:
        print(f"{key}: {value:.4f}")
```
```
==================================================
평가 결과
==================================================
eval_loss: 0.2510
eval_exact_match: 0.5747
eval_micro_f1: 0.9485
eval_macro_f1: 0.9479
eval_linguistic_acceptability_f1: 0.9317
eval_consistency_f1: 0.9448
eval_interestingness_f1: 0.9600
eval_unbias_f1: 0.9619
eval_harmlessness_f1: 0.9646
eval_no_hallucination_f1: 0.9121
eval_understandability_f1: 0.9499
eval_sensibleness_f1: 0.9377
eval_specificity_f1: 0.9686

```
- Macro F1 (0.9479) & Micro F1 (0.9485): 9개의 지표 중 어느 하나가 크게 뒤처지지 않고 <strong>모델이 모든 기준을 골고루 잘 학습했다.</strong>
- Exact Match (0.5747): 9개 항목을 단 하나도 틀리지 않고 모두 맞춘 비율. 멀티 레이블 분류의 특성상 9개를 동시에 다 맞히는 것은 확률적으로 매우 어려움($0.94^9 \approx 0.57$) <strong>즉, 현재 모델은 통계적으로 기대할 수 있는 최상의 정확도</strong>를 보여주고 있다.
- ``no_hallucination_f1``점수가 가장 낮다는 점은 "사실 관계 확인의 어려움"때문이다. 진짜 정보를 파악하게 하려면?
-> Contrastive Learning을 이 지표에 집중적으로 적용하는 것도 하나의 방법.

```
# 추론 테스트
sample_question = "한국의 수도는 어디야?"
sample_response = "한국의 수도는 부산입니다. 서울은 대한민국의 정치, 경제, 문화의 중심지로, 약 1000만 명의 인구가 거주하고 있습니다."
sample_input = f"{sample_question} [SEP] {sample_response}"

print(f"질문: {sample_question}")
print(f"응답: {sample_response}")
print("\n" + "=" * 50)
print("예측 결과")
print("=" * 50)

results = predict(sample_input, model, tokenizer, device)
for criterion, values in results.items():
    status = "✓" if values['prediction'] == 1 else "✗"
    print(f"{status} {criterion}: {values['probability']:.2%}")
```
```
질문: 한국의 수도는 어디야?
응답: 한국의 수도는 서울입니다. 서울은 대한민국의 정치, 경제, 문화의 중심지로, 약 1000만 명의 인구가 거주하고 있습니다.

==================================================
예측 결과
==================================================
✓ linguistic_acceptability: 97.49%
✓ consistency: 99.19%
✓ interestingness: 98.21%
✓ unbias: 99.58%
✓ harmlessness: 99.45%
✓ no_hallucination: 97.56%
✓ understandability: 98.70%
✓ sensibleness: 98.41%
✓ specificity: 98.65%
```
- 모델이 상식적인 수준에서 아주 잘 작동하고 있다.

<strong>모델이 너무 'Overconfidence'</strong>한 것은 아닐까?
-> 한국의 수도는 '책상'인 것처럼 수정
```
질문: 한국의 수도는 어디야?
응답: 한국의 수도는 책상입니다. 책상은 지리적으로 대한민국의 정치, 경제, 문화의 중심지로, 약 1000만 명의 인구가 거주하고 있습니다.

==================================================
예측 결과
==================================================
✓ linguistic_acceptability: 96.67%
✓ consistency: 98.24%
✓ interestingness: 98.13%
✓ unbias: 99.62%
✓ harmlessness: 99.50%
✓ no_hallucination: 94.99%
✓ understandability: 98.89%
✓ sensibleness: 97.08%
✓ specificity: 98.64%
```
-> 현재 모델이 사실 여부가 아니라 문장의 그럴싸함에 완전히 매몰되어 있음을 알수있음.
--> 그렇다면 어떻게 해결해야 할까?
- 1. Hard Negative 추가
- 2. Soft Labels
- 3. Contrasive Learning

### 1. Hard Negative 추가
```
추론 테스트
sample_question = "한국의 수도는 어디야?"
sample_response = "한국의 수도는 의자입니다. 의자는 지리적으로 대한민국의 정치, 경제, 문화의 중심지로, 약 1000만 명의 인구가 거주하고 있습니다."
```

```
==================================================
예측 결과
==================================================
✓ linguistic_acceptability: 98.91%
✓ consistency: 70.48%
✓ interestingness: 94.29%
✓ unbias: 99.91%
✓ harmlessness: 99.95%
✗ no_hallucination: 22.92%
✓ understandability: 98.66%
✓ sensibleness: 97.12%
✓ specificity: 96.74%
```

```
추론 테스트
sample_question = "한국의 수도는 어디야?"
sample_response = "한국의 수도는 연필입니다. 연필은 지리적으로 대한민국의 정치, 경제, 문화의 중심지로, 약 1000만 명의 인구가 거주하고 있습니다."
```
```
==================================================
예측 결과
==================================================
✓ linguistic_acceptability: 98.43%
✗ consistency: 34.05%
✓ interestingness: 86.36%
✓ unbias: 99.73%
✓ harmlessness: 99.83%
✗ no_hallucination: 3.75%
✓ understandability: 98.19%
✓ sensibleness: 91.66%
✓ specificity: 91.49%
```
-> 문맥상 전혀 어울리지 않는 단어가 나오는 경우, 90% 이상의 높은 정확도로 잡아내고 있음.