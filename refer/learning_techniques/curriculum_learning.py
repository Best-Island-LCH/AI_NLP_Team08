"""
Curriculum Learning for AI Quality Evaluation

커리큘럼 학습을 적용한 AI 품질 평가
- 난이도 기반 샘플 정렬
- 에폭별 점진적 학습
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, Sampler, DataLoader
from typing import List, Dict, Optional
from sklearn.metrics import f1_score


class CurriculumDataset(Dataset):
    """
    커리큘럼 학습을 위한 데이터셋
    각 샘플의 난이도 정보를 포함
    """
    
    CRITERIA = [
        'linguistic_acceptability', 'consistency', 'interestingness',
        'unbias', 'harmlessness', 'no_hallucination',
        'understandability', 'sensibleness', 'specificity'
    ]
    
    def __init__(self, samples: List[dict], tokenizer, max_length: int = 512):
        """
        Args:
            samples: 샘플 리스트 (dict 형태)
            tokenizer: HuggingFace 토크나이저
            max_length: 최대 시퀀스 길이
        """
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 각 샘플의 난이도 계산
        self.difficulties = self._compute_difficulties()
    
    def _compute_difficulties(self) -> np.ndarray:
        """평가자 일치도 기반 난이도 계산"""
        difficulties = []
        
        for sample in self.samples:
            criterion_difficulties = []
            
            for c in self.CRITERIA:
                unanimous_key = f'{c}_unanimous'
                if unanimous_key in sample:
                    # 만장일치가 아니면 어려움
                    criterion_difficulties.append(
                        0 if sample[unanimous_key] == 1 else 1
                    )
                else:
                    # unanimous 정보 없으면 기본값
                    criterion_difficulties.append(0.5)
            
            # 평균 난이도
            difficulty = np.mean(criterion_difficulties)
            difficulties.append(difficulty)
        
        return np.array(difficulties)
    
    def get_easy_indices(self, threshold: float = 0.3) -> np.ndarray:
        """쉬운 샘플 인덱스 (난이도 <= threshold)"""
        return np.where(self.difficulties <= threshold)[0]
    
    def get_medium_indices(self, low: float = 0.3, high: float = 0.7) -> np.ndarray:
        """중간 난이도 샘플 인덱스"""
        return np.where(
            (self.difficulties > low) & (self.difficulties <= high)
        )[0]
    
    def get_hard_indices(self, threshold: float = 0.7) -> np.ndarray:
        """어려운 샘플 인덱스 (난이도 > threshold)"""
        return np.where(self.difficulties > threshold)[0]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        
        encoding = self.tokenizer(
            sample['input_text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # _majority 접미사가 있는 경우 처리
        labels = []
        for c in self.CRITERIA:
            if f'{c}_majority' in sample:
                labels.append(sample[f'{c}_majority'])
            elif c in sample:
                labels.append(sample[c])
            else:
                labels.append(0)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(labels, dtype=torch.float),
            'difficulty': self.difficulties[idx]
        }


class CurriculumSampler(Sampler):
    """
    커리큘럼 학습을 위한 샘플러
    에폭에 따라 포함할 샘플 결정
    """
    
    def __init__(
        self, 
        dataset: CurriculumDataset, 
        total_epochs: int,
        strategy: str = 'linear'
    ):
        """
        Args:
            dataset: CurriculumDataset 인스턴스
            total_epochs: 총 학습 에폭 수
            strategy: 커리큘럼 전략 ('linear', 'sqrt', 'step')
        """
        self.dataset = dataset
        self.total_epochs = total_epochs
        self.strategy = strategy
        self.current_epoch = 0
        
        # 난이도별 인덱스
        self.easy_indices = dataset.get_easy_indices()
        self.medium_indices = dataset.get_medium_indices()
        self.hard_indices = dataset.get_hard_indices()
        
        print(f"📊 난이도 분포:")
        print(f"   Easy: {len(self.easy_indices)} ({len(self.easy_indices)/len(dataset)*100:.1f}%)")
        print(f"   Medium: {len(self.medium_indices)} ({len(self.medium_indices)/len(dataset)*100:.1f}%)")
        print(f"   Hard: {len(self.hard_indices)} ({len(self.hard_indices)/len(dataset)*100:.1f}%)")
    
    def set_epoch(self, epoch: int):
        """현재 에폭 설정 (학습 루프에서 호출)"""
        self.current_epoch = epoch
    
    def _get_competence(self) -> float:
        """
        현재 에폭의 역량(competence) 계산
        역량에 따라 포함할 난이도 결정
        """
        if self.strategy == 'linear':
            # 선형 증가
            return min(1.0, (self.current_epoch + 1) / self.total_epochs)
        elif self.strategy == 'sqrt':
            # 제곱근 (초기에 빠르게, 후기에 느리게)
            return min(1.0, np.sqrt((self.current_epoch + 1) / self.total_epochs))
        elif self.strategy == 'step':
            # 단계적
            if self.current_epoch < self.total_epochs // 3:
                return 0.33
            elif self.current_epoch < 2 * self.total_epochs // 3:
                return 0.66
            else:
                return 1.0
        else:
            return 1.0
    
    def __iter__(self):
        competence = self._get_competence()
        
        # 역량에 따라 포함할 샘플 결정
        if competence < 0.33:
            # 쉬운 샘플만
            indices = self.easy_indices.copy()
        elif competence < 0.66:
            # 쉬운 + 중간
            indices = np.concatenate([self.easy_indices, self.medium_indices])
        else:
            # 모든 샘플
            indices = np.arange(len(self.dataset))
        
        # 셔플
        np.random.shuffle(indices)
        
        print(f"   Epoch {self.current_epoch}: competence={competence:.2f}, "
              f"samples={len(indices)}/{len(self.dataset)}")
        
        return iter(indices.tolist())
    
    def __len__(self):
        competence = self._get_competence()
        
        if competence < 0.33:
            return len(self.easy_indices)
        elif competence < 0.66:
            return len(self.easy_indices) + len(self.medium_indices)
        else:
            return len(self.dataset)


class CurriculumTrainer:
    """
    커리큘럼 학습 트레이너
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        train_dataset: CurriculumDataset, 
        val_loader: DataLoader, 
        device: torch.device, 
        config: dict
    ):
        """
        Args:
            model: PyTorch 모델
            train_dataset: CurriculumDataset 인스턴스
            val_loader: 검증 데이터로더
            device: 학습 디바이스
            config: 학습 설정
                - epochs: 총 에폭 수
                - batch_size: 배치 크기
                - learning_rate: 학습률
                - curriculum_strategy: 커리큘럼 전략
        """
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # 커리큘럼 샘플러
        self.sampler = CurriculumSampler(
            train_dataset,
            total_epochs=config['epochs'],
            strategy=config.get('curriculum_strategy', 'linear')
        )
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['learning_rate']
        )
        
        self.criterion = nn.BCEWithLogitsLoss()
    
    def train(self):
        """전체 학습"""
        best_f1 = 0
        
        print(f"\n🎓 Curriculum Learning 시작")
        print(f"   전략: {self.config.get('curriculum_strategy', 'linear')}")
        print(f"   에폭: {self.config['epochs']}")
        
        for epoch in range(self.config['epochs']):
            # 샘플러에 현재 에폭 알림
            self.sampler.set_epoch(epoch)
            
            # 데이터로더 생성 (매 에폭마다 새로)
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.config['batch_size'],
                sampler=self.sampler
            )
            
            # 학습
            train_loss = self._train_epoch(train_loader)
            
            # 평가
            val_results = self._evaluate()
            
            print(f"   Train Loss: {train_loss:.4f} | Val F1: {val_results['f1_macro']:.4f}")
            
            if val_results['f1_macro'] > best_f1:
                best_f1 = val_results['f1_macro']
                torch.save(self.model.state_dict(), 'best_curriculum_model.pt')
                print("   ✅ Best model saved!")
        
        print(f"\n🏆 Best F1: {best_f1:.4f}")
        return best_f1
    
    def _train_epoch(self, loader: DataLoader) -> float:
        """한 에폭 학습"""
        self.model.train()
        total_loss = 0
        
        for batch in loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            self.optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask)
            loss = self.criterion(logits, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(loader)
    
    def _evaluate(self) -> dict:
        """검증 데이터 평가"""
        self.model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels']
                
                logits = self.model(input_ids, attention_mask)
                preds = (torch.sigmoid(logits) > 0.5).float()
                
                all_preds.append(preds.cpu())
                all_labels.append(labels)
        
        preds = torch.cat(all_preds).numpy()
        labels = torch.cat(all_labels).numpy()
        
        return {
            'f1_macro': f1_score(labels, preds, average='macro', zero_division=0)
        }


# ============================================================
# 사용 예시
# ============================================================

def example_usage():
    """Curriculum Learning 사용 예시"""
    
    print("Curriculum Learning 예시")
    print("=" * 50)
    
    # 더미 샘플 생성
    samples = []
    for i in range(100):
        # Easy 샘플 (만장일치)
        if i < 40:
            sample = {
                'input_text': f'쉬운 샘플 {i}',
                'linguistic_acceptability_majority': 1,
                'linguistic_acceptability_unanimous': 1,
                'consistency_majority': 1,
                'consistency_unanimous': 1,
                # ... 나머지 기준들도 unanimous=1
            }
            for c in CurriculumDataset.CRITERIA:
                sample[f'{c}_majority'] = 1
                sample[f'{c}_unanimous'] = 1
        # Hard 샘플 (불일치)
        else:
            sample = {
                'input_text': f'어려운 샘플 {i}',
            }
            for c in CurriculumDataset.CRITERIA:
                sample[f'{c}_majority'] = np.random.randint(0, 2)
                sample[f'{c}_unanimous'] = 0  # 불일치
        
        samples.append(sample)
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('klue/roberta-base')
    
    # 데이터셋 생성
    dataset = CurriculumDataset(samples, tokenizer)
    
    # 샘플러 테스트
    sampler = CurriculumSampler(
        dataset,
        total_epochs=10,
        strategy='sqrt'
    )
    
    print("\n에폭별 샘플 수:")
    for epoch in range(10):
        sampler.set_epoch(epoch)
        indices = list(sampler)
        # (출력은 __iter__에서 자동으로)


if __name__ == "__main__":
    example_usage()
