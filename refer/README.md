# 📚 고급 NLP 기법 레퍼런스

> 이 폴더는 AI 품질 평가 태스크를 위한 고급 NLP 기법들을 정리한 레퍼런스입니다.  
> 각 기법마다 **개념 설명 (.md)** + **구현 코드 (.py)** 로 구성되어 있습니다.

---

## 📁 폴더 구조

```
refer/
├── README.md                      # 이 파일
├── 00_decision_guide.md           # ⭐ 의사결정 가이드 (먼저 읽기!)
├── 01_model_comparison.md         # 사전학습 모델 비교 가이드
│
├── learning_techniques/           # 🎓 고급 학습 기법
│   ├── contrastive_learning.md    # Contrastive Learning 개념
│   ├── contrastive_learning.py    # 구현 코드
│   ├── multitask_learning.md      # Multi-task Learning 개념
│   ├── multitask_learning.py      # 구현 코드
│   ├── curriculum_learning.md     # Curriculum Learning 개념
│   ├── curriculum_learning.py     # 구현 코드
│   ├── uncertainty_modeling.md    # 평가자 불일치 모델링 개념
│   └── uncertainty_modeling.py    # 구현 코드
│
├── architectures/                 # 🏗️ 고급 모델 아키텍처
│   ├── hierarchical_attention.md  # 계층적 어텐션 개념
│   ├── hierarchical_attention.py  # 구현 코드
│   ├── cross_encoder.md           # Cross-Encoder 개념
│   ├── cross_encoder.py           # 구현 코드
│   ├── multihead_classification.md # Multi-Head 분류 개념
│   └── multihead_classification.py # 구현 코드
│
├── team_experiments/              # 👥 팀원 실험 기록
│   ├── README.md                  # 기여 가이드라인
│   └── TEMPLATE.md                # 실험 기록 템플릿
│
└── experiment_template.md         # 📊 실험 설계 템플릿
```

---

## ⭐ 시작점: 의사결정 가이드

> **어디서부터 시작해야 할지 모르겠다면?**
>
> 👉 [00_decision_guide.md](00_decision_guide.md)를 먼저 읽어보세요!

이 가이드는 다음 질문에 답합니다:
- 현재 모델의 문제점이 무엇인가?
- 어떤 기법을 적용해야 하는가?
- 어떤 순서로 시도해야 하는가?

---

## 🚀 빠른 시작

### 0️⃣ 문제 진단 (필수!)
먼저 [00_decision_guide.md](00_decision_guide.md)를 읽고 현재 모델의 문제점을 파악하세요.

### 1️⃣ 모델 선택
[01_model_comparison.md](01_model_comparison.md)를 읽고 베이스 모델을 선택하세요.

### 2️⃣ 학습 기법 선택
성능 향상을 위해 다음 중 하나를 적용해보세요:

| 기법 | 추천 상황 | 난이도 |
|------|----------|-------|
| [Soft Labels](learning_techniques/uncertainty_modeling.md) | 가장 먼저 시도 | ⭐ |
| [Curriculum Learning](learning_techniques/curriculum_learning.md) | 학습 안정성 필요 | ⭐⭐ |
| [Multi-task Learning](learning_techniques/multitask_learning.md) | 기준 간 관계 활용 | ⭐⭐ |
| [Contrastive Learning](learning_techniques/contrastive_learning.md) | 임베딩 품질 향상 | ⭐⭐⭐ |

### 3️⃣ 아키텍처 선택 (선택적)
기본 아키텍처로 부족하면:

| 아키텍처 | 추천 상황 | 난이도 |
|----------|----------|-------|
| [Multi-Head](architectures/multihead_classification.md) | 기준별 특화 필요 | ⭐⭐ |
| [Cross-Encoder](architectures/cross_encoder.md) | Q-A 대응관계 중요 | ⭐⭐⭐ |
| [Hierarchical](architectures/hierarchical_attention.md) | 긴 대화 처리 | ⭐⭐⭐ |

---

## 💡 권장 학습 순서

```
초보자:
  1. 00_decision_guide.md → 문제 진단 방법
  2. 01_model_comparison.md → 모델 이해
  3. uncertainty_modeling.md → Soft Labels 적용
  4. multihead_classification.md → 기준별 분류

중급자:
  1. curriculum_learning.md → 커리큘럼 학습
  2. multitask_learning.md → 다중 태스크 학습
  3. cross_encoder.md → 상호작용 모델링

고급자:
  1. contrastive_learning.md → 대조 학습
  2. hierarchical_attention.md → 계층적 모델
  3. 기법들 조합하여 실험
  4. team_experiments/에 결과 공유
```

---

## 📝 각 파일 사용법

### 개념 문서 (.md)
- **왜 필요한가?**: 문제점과 해결책
- **어떻게 작동하는가?**: 원리 설명
- **이 태스크에 어떻게 적용하는가?**: 구체적 적용 방법
- **예상 효과 및 주의사항**: 기대 효과와 주의점

### 구현 코드 (.py)
- 바로 실행 가능한 전체 코드
- 주석으로 상세 설명 포함
- 사용 예시 함수 포함

```python
# 예시: Contrastive Learning 사용
from learning_techniques.contrastive_learning import ContrastiveQualityModel

model = ContrastiveQualityModel(
    model_name='klue/roberta-base',
    num_labels=9
)
```

---

## 👥 팀원 기여 공간

새로운 기법이나 실험 결과를 발견했다면 [team_experiments/](team_experiments/)에 공유해주세요!

- 📋 [기여 가이드라인](team_experiments/README.md)
- 📝 [실험 기록 템플릿](team_experiments/TEMPLATE.md)

### 기여 방법
1. `TEMPLATE.md`를 복사
2. 파일명: `[날짜]_[이름]_[기법명].md`
3. 내용 작성 후 커밋

---

## 🔗 관련 문서

### 프로젝트 문서
- [프로젝트 README](../README.md) - 전체 구조 및 시작 가이드
- [01_implementation_strategy.md](../01_implementation_strategy.md) - 베이스라인 구현
- [02_advanced_strategy.md](../02_advanced_strategy.md) - 고급 전략 개요

### 최적화
- [optimization/](../optimization/) - 모델 최적화 (Distillation, Quantization 등)

### 실험 도구
- [experiment_template.md](experiment_template.md) - 실험 설계 템플릿
