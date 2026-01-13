# 실험 설계 템플릿

> 체계적인 실험 진행을 위한 가이드라인

---

## 1. 실험 설정 템플릿

```python
# config.py

EXPERIMENTS = {
    'baseline': {
        'model': 'klue/roberta-base',
        'architecture': 'simple',  # [CLS] + Linear
        'learning_rate': 2e-5,
        'batch_size': 16,
        'epochs': 5,
    },
    'contrastive': {
        'model': 'klue/roberta-base',
        'architecture': 'contrastive',
        'lambda_contrastive': 0.1,
        'temperature': 0.07,
        'learning_rate': 2e-5,
        'batch_size': 32,  # 크게
        'epochs': 5,
    },
    'multitask': {
        'model': 'klue/roberta-base',
        'architecture': 'multitask',
        'weighting': 'uncertainty',
        'use_interaction': True,
        'learning_rate': 2e-5,
        'batch_size': 16,
        'epochs': 5,
    },
    'hierarchical': {
        'model': 'klue/roberta-base',
        'architecture': 'hierarchical',
        'max_turns': 8,
        'learning_rate': 1e-5,  # 더 낮게
        'batch_size': 8,
        'epochs': 5,
    },
    'cross_encoder': {
        'model': 'klue/roberta-base',
        'architecture': 'cross_encoder',
        'num_cross_layers': 2,
        'learning_rate': 1e-5,
        'batch_size': 8,
        'epochs': 5,
    },
    'deberta': {
        'model': 'microsoft/deberta-v3-base',
        'architecture': 'simple',
        'learning_rate': 1e-5,
        'batch_size': 8,
        'epochs': 5,
    },
}
```

---

## 2. 평가 지표

### 코드

```python
from sklearn.metrics import (
    f1_score, precision_score, recall_score, 
    roc_auc_score, accuracy_score
)
import numpy as np

def comprehensive_evaluation(y_true, y_pred, y_prob):
    """
    종합 평가 지표 계산
    
    Args:
        y_true: 실제 라벨 [N, 9]
        y_pred: 예측 라벨 [N, 9] (0 or 1)
        y_prob: 예측 확률 [N, 9] (0~1)
    
    Returns:
        results: 평가 결과 딕셔너리
    """
    results = {
        # 전체 메트릭
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'accuracy': accuracy_score(y_true.flatten(), y_pred.flatten()),
    }
    
    # AUC (클래스별로 계산 후 평균)
    try:
        results['auc_macro'] = roc_auc_score(y_true, y_prob, average='macro')
    except:
        results['auc_macro'] = 0.0
    
    # 기준별 메트릭
    criteria = [
        'linguistic', 'consistency', 'interesting',
        'unbias', 'harmless', 'no_halluc',
        'understand', 'sensible', 'specific'
    ]
    
    results['per_criterion'] = {}
    for i, c in enumerate(criteria):
        results['per_criterion'][c] = {
            'f1': f1_score(y_true[:, i], y_pred[:, i], zero_division=0),
            'precision': precision_score(y_true[:, i], y_pred[:, i], zero_division=0),
            'recall': recall_score(y_true[:, i], y_pred[:, i], zero_division=0),
        }
        try:
            results['per_criterion'][c]['auc'] = roc_auc_score(y_true[:, i], y_prob[:, i])
        except:
            results['per_criterion'][c]['auc'] = 0.0
    
    return results
```

### 주요 지표 설명

| 지표 | 설명 | 우선순위 |
|-----|------|---------|
| **F1 Macro** | 기준별 F1의 평균 | ⭐⭐⭐ 주요 |
| **F1 Micro** | 전체 TP/FP/FN 기반 F1 | ⭐⭐ |
| **AUC** | 순위 기반 평가 | ⭐⭐ |
| **Accuracy** | 정확도 | ⭐ (불균형 시 부적합) |

---

## 3. 실험 체크리스트

### 실험 전
- [ ] 데이터 로드 확인
- [ ] Train/Val/Test 분할 확인
- [ ] GPU 메모리 확인
- [ ] 랜덤 시드 고정
- [ ] 베이스라인 결과 기록

### 실험 중
- [ ] Loss 수렴 확인
- [ ] Validation 성능 모니터링
- [ ] GPU 사용률 확인
- [ ] 메모리 누수 확인

### 실험 후
- [ ] 최종 성능 기록
- [ ] 기준별 성능 분석
- [ ] 실패 케이스 분석
- [ ] 재현 가능성 확인

---

## 4. 결과 보고 템플릿

```markdown
# 실험 결과 보고서

## 실험 정보
- 실험자: [이름]
- 날짜: [YYYY-MM-DD]
- 목적: [실험 목적]

## 1. 실험 설정
| 설정 | 값 |
|------|-----|
| 모델 | klue/roberta-base |
| 아키텍처 | [아키텍처명] |
| Learning Rate | 2e-5 |
| Batch Size | 16 |
| Epochs | 5 |

## 2. 전체 성능 비교
| Model | F1 (macro) | F1 (micro) | AUC | 학습시간 |
|-------|------------|------------|-----|----------|
| Baseline | 0.XXX | 0.XXX | 0.XX | XX min |
| 실험모델 | 0.XXX | 0.XXX | 0.XX | XX min |
| **Δ** | +X.X% | +X.X% | +X.X% | |

## 3. 기준별 성능 (F1)
| Criterion | Baseline | 실험모델 | Δ |
|-----------|----------|----------|---|
| linguistic | 0.XXX | 0.XXX | +X.X% |
| consistency | 0.XXX | 0.XXX | +X.X% |
| interesting | 0.XXX | 0.XXX | +X.X% |
| unbias | 0.XXX | 0.XXX | +X.X% |
| harmless | 0.XXX | 0.XXX | +X.X% |
| no_halluc | 0.XXX | 0.XXX | +X.X% |
| understand | 0.XXX | 0.XXX | +X.X% |
| sensible | 0.XXX | 0.XXX | +X.X% |
| specific | 0.XXX | 0.XXX | +X.X% |

## 4. 분석
### 성공 요인
- [요인 1]
- [요인 2]

### 실패 요인 / 한계
- [한계 1]
- [한계 2]

## 5. 결론 및 다음 단계
- 결론: [요약]
- 다음 단계: [제안]
```

---

## 5. 실험 실행 스크립트

```python
# run_experiment.py

import torch
import json
import time
from datetime import datetime

def run_experiment(config_name, config):
    """실험 실행"""
    
    print(f"\n{'='*60}")
    print(f"실험: {config_name}")
    print(f"{'='*60}")
    
    results = {
        'config_name': config_name,
        'config': config,
        'start_time': datetime.now().isoformat(),
    }
    
    start_time = time.time()
    
    try:
        # 1. 모델 생성
        model = create_model(config)
        
        # 2. 데이터 로드
        train_loader, val_loader, test_loader = load_data(config)
        
        # 3. 학습
        best_model = train(model, train_loader, val_loader, config)
        
        # 4. 평가
        test_results = evaluate(best_model, test_loader)
        
        results['status'] = 'success'
        results['metrics'] = test_results
        
    except Exception as e:
        results['status'] = 'failed'
        results['error'] = str(e)
    
    results['duration_minutes'] = (time.time() - start_time) / 60
    results['end_time'] = datetime.now().isoformat()
    
    # 결과 저장
    filename = f"results/{config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n결과 저장: {filename}")
    
    return results


def run_all_experiments():
    """모든 실험 실행"""
    
    all_results = []
    
    for config_name, config in EXPERIMENTS.items():
        result = run_experiment(config_name, config)
        all_results.append(result)
    
    # 비교 테이블 출력
    print_comparison_table(all_results)
    
    return all_results
```

---

## 6. 팁

### 빠른 실험을 위한 팁

```
1. 샘플 데이터로 먼저 테스트
   - 1000개 샘플로 파이프라인 검증
   - 1 에폭만 학습

2. 점진적 복잡도 증가
   - Baseline → Soft Labels → Multi-task → ...
   - 각 단계에서 성능 확인

3. 조기 종료 활용
   - patience=3 정도로 설정
   - validation loss 기준

4. 체크포인트 저장
   - 매 에폭 저장
   - 최고 성능 모델 따로 저장
```

### 실험 기록 유지

```
experiments/
├── 2024-01-15_baseline/
│   ├── config.json
│   ├── logs/
│   ├── checkpoints/
│   └── results.json
├── 2024-01-16_contrastive/
│   ├── ...
└── comparison.md
```
