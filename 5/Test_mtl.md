# Test_mtl

# 1차 시도

hard-sharing

batch size 64

epoch 3

```python
학습 시작...
 [18777/18777 58:06, Epoch 3/3]
Epoch	Training Loss	Validation Loss	Exact Match	Micro F1	Macro F1	Linguistic Acceptability Acc	Consistency Acc	Interestingness Acc	Unbias Acc	Harmlessness Acc	No Hallucination Acc	Understandability Acc	Sensibleness Acc	Specificity Acc
1	1.260700	1.292921	0.710392	0.972526	0.972379	0.967950	0.953803	0.942614	0.973525	0.987112	0.904270	0.926089	0.949288	0.955802
2	1.112100	1.241228	0.722521	0.973646	0.973509	0.970947	0.956181	0.943153	0.974304	0.987612	0.909185	0.929247	0.952505	0.956361
3	1.041600	1.256541	0.721722	0.973623	0.973490	0.971207	0.956401	0.942894	0.975183	0.987472	0.909625	0.927808	0.953064	0.955961
학습 완료!
```

```python
==================================================
평가 결과
==================================================
eval_loss: 1.2412
eval_exact_match: 0.7225
eval_micro_f1: 0.9736
eval_macro_f1: 0.9735
```

```python
==================================================
기준별 정확도
==================================================
linguistic_acceptability: 0.9709
consistency: 0.9562
interestingness: 0.9432
unbias: 0.9743
harmlessness: 0.9876
no_hallucination: 0.9092
understandability: 0.9292
sensibleness: 0.9525
specificity: 0.9564
```

# 2차 시도

Batch size 128

Epoch 6 (Early stopping)

```python
Epoch	Training Loss	Validation Loss	Exact Match	Micro F1	Macro F1	Linguistic Acceptability Acc	Consistency Acc	Interestingness Acc	Unbias Acc	Harmlessness Acc	No Hallucination Acc	Understandability Acc	Sensibleness Acc	Specificity Acc
1	1.334500	1.368648	0.693508	0.970592	0.970368	0.965992	0.950527	0.942454	0.969269	0.984055	0.892421	0.923632	0.946650	0.955602
2	1.278100	1.289501	0.708574	0.972221	0.972072	0.968330	0.952225	0.942394	0.971687	0.986293	0.903810	0.925390	0.949827	0.955582
3	1.165900	1.312926	0.709973	0.972527	0.972354	0.970488	0.953743	0.942754	0.973625	0.987092	0.903750	0.923812	0.950666	0.955961
4	1.068200	1.352274	0.711012	0.972276	0.972139	0.970688	0.952165	0.940975	0.974064	0.986772	0.906008	0.923812	0.948968	0.954962
5	0.917300	1.378835	0.708434	0.972172	0.972064	0.971027	0.954443	0.939697	0.974284	0.986553	0.904390	0.922713	0.948628	0.954383
6	0.815800	1.520931	0.705117	0.971774	0.971623	0.971407	0.953104	0.938658	0.971667	0.987052	0.904949	0.920754	0.949088	0.953564

```

```python
==================================================
평가 결과
==================================================
eval_loss: 1.3129
eval_exact_match: 0.7100
eval_micro_f1: 0.9725
eval_macro_f1: 0.9724
```

```python
==================================================
기준별 정확도
==================================================
linguistic_acceptability: 0.9705
consistency: 0.9537
interestingness: 0.9428
unbias: 0.9736
harmlessness: 0.9871
no_hallucination: 0.9038
understandability: 0.9238
sensibleness: 0.9507
specificity: 0.9560
```

<aside>
<img src="/icons/book_gray.svg" alt="/icons/book_gray.svg" width="40px" />

IDEA 

전체 9개 헤드에 대한 Interaction Layer을 주는 게 맞을까?

EDA 에 따르면 낮은 상관관계를 보이는 라벨들도 있는데 MTL은 전체를 유기적으로 봄. 

그렇다면 정보 오염(Noise)/학습 난이도 상승 일어나는거 아님?

→특정 과목의 성능이 안좋다면 라벨별로 묶어서 Interaction Layer 주기(Multi-head)

→Validation Loss가 꾸준히 줄어든다면 이부분은 수정하지 않아도 될듯

*Training Loss 만 낮아지면 Overfitting

*Validation Loss 가 낮아지길 제발

→2차 시도 이후 epoch3 넘어갈 시 과적합 문제 발생, epoch3로 주되 다른 요소를 바꿔보기

EDA Data

**높은 상관관계 (그룹화 가능):**

| **그룹** | **기준** | **상관계수** |
| --- | --- | --- |
| Content Quality | interestingness ↔ specificity | **0.909** |
| Safety | unbias ↔ harmlessness | **0.817** |
| Coherence | consistency ↔ no_hallucination | **0.619** |

**낮은 상관관계 (독립적):**

- linguistic_acceptability ↔ consistency: 0.014
- unbias ↔ understandability: -0.002

**시사점:**

- **Multi-Head 모델**에서 그룹별 헤드 구성 가능
- 상관관계 높은 기준끼리 **Interaction Layer** 적용 고려
</aside>