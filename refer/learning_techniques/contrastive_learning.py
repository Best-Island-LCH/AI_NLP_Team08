"""
Contrastive Learning for AI Quality Evaluation

대조 학습을 적용한 AI 품질 평가 모델
- Supervised Contrastive Loss
- BCE Classification Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class ContrastiveQualityModel(nn.Module):
    """
    Contrastive Learning을 적용한 AI 품질 평가 모델
    
    구조:
      Input → BERT → [CLS] → Projection Head → Contrastive Loss
                           ↘ Classification Head → BCE Loss
    """
    
    def __init__(
        self, 
        model_name: str = 'klue/roberta-base', 
        num_labels: int = 9, 
        projection_dim: int = 256, 
        temperature: float = 0.07
    ):
        """
        Args:
            model_name: HuggingFace 모델 ID
            num_labels: 분류 라벨 수 (기본 9개 기준)
            projection_dim: Projection Head 출력 차원
            temperature: Contrastive Loss의 temperature
        """
        super().__init__()
        
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        
        # Projection Head (contrastive learning용)
        # 논문에서는 2-layer MLP가 효과적이라고 함
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, projection_dim)
        )
        
        # Classification Head (분류용)
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_labels)
        )
        
        self.temperature = temperature
        self.num_labels = num_labels
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        return_projection: bool = False
    ):
        """
        순전파
        
        Args:
            input_ids: 토큰 ID [batch_size, seq_len]
            attention_mask: 어텐션 마스크 [batch_size, seq_len]
            return_projection: projection 출력 반환 여부
        
        Returns:
            logits: 분류 로짓 [batch_size, num_labels]
            projection: (optional) 대조 학습용 임베딩 [batch_size, projection_dim]
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] 토큰
        
        # 분류 로짓
        logits = self.classifier(cls_output)
        
        if return_projection:
            # L2 정규화된 projection
            projection = self.projection_head(cls_output)
            projection = F.normalize(projection, p=2, dim=1)
            return logits, projection
        
        return logits
    
    def contrastive_loss(
        self, 
        z_i: torch.Tensor, 
        z_j: torch.Tensor, 
        labels_i: torch.Tensor, 
        labels_j: torch.Tensor
    ) -> torch.Tensor:
        """
        Supervised Contrastive Loss
        
        같은 라벨을 가진 샘플끼리 positive pair로 취급
        
        Args:
            z_i, z_j: projection 임베딩 [batch_size, projection_dim]
            labels_i, labels_j: 라벨 [batch_size, num_labels]
        
        Returns:
            loss: Contrastive loss 스칼라
        """
        batch_size = z_i.size(0)
        
        # 코사인 유사도 계산
        # z_i와 z_j를 concat하여 전체 유사도 행렬 계산
        z = torch.cat([z_i, z_j], dim=0)  # [2*batch_size, projection_dim]
        sim_matrix = torch.mm(z, z.t()) / self.temperature  # [2*batch_size, 2*batch_size]
        
        # 라벨 매칭 (positive pair 마스크)
        labels = torch.cat([labels_i, labels_j], dim=0)  # [2*batch_size, num_labels]
        
        # 같은 라벨을 가진 쌍 찾기 (multi-label이므로 모든 라벨이 같아야 positive)
        label_match = torch.mm(labels, labels.t())  # [2*batch_size, 2*batch_size]
        label_sum = labels.sum(dim=1, keepdim=True)
        positive_mask = (label_match == label_sum) & (label_match == label_sum.t())
        
        # 자기 자신 제외
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        positive_mask = positive_mask & ~mask
        
        # InfoNCE Loss 계산
        exp_sim = torch.exp(sim_matrix)
        
        # 분모: 자기 자신 제외한 모든 샘플
        denominator = exp_sim.masked_fill(mask, 0).sum(dim=1)
        
        # 분자: positive pair들의 합
        numerator = (exp_sim * positive_mask.float()).sum(dim=1)
        
        # 로그 손실 (positive가 있는 샘플만)
        positive_count = positive_mask.sum(dim=1)
        loss = -torch.log(numerator / denominator + 1e-8)
        loss = loss[positive_count > 0].mean()
        
        return loss


class ContrastiveTrainer:
    """
    Contrastive Learning 학습기
    """
    
    def __init__(self, model, train_loader, val_loader, device, config):
        """
        Args:
            model: ContrastiveQualityModel
            train_loader: 학습 데이터로더
            val_loader: 검증 데이터로더
            device: 학습 디바이스
            config: 학습 설정 딕셔너리
                - learning_rate: 학습률
                - lambda_contrastive: Contrastive loss 가중치
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config['learning_rate']
        )
        
        # 손실 함수
        self.ce_criterion = nn.BCEWithLogitsLoss()
        
        # 손실 가중치
        self.lambda_contrastive = config.get('lambda_contrastive', 0.1)
    
    def train_epoch(self):
        """
        한 에폭 학습
        
        Returns:
            dict: 손실 정보 {'total_loss', 'ce_loss', 'contrastive_loss'}
        """
        self.model.train()
        total_loss = 0
        total_ce_loss = 0
        total_cl_loss = 0
        
        for batch in self.train_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward (projection 포함)
            logits, projection = self.model(
                input_ids, attention_mask, return_projection=True
            )
            
            # 분류 손실
            ce_loss = self.ce_criterion(logits, labels)
            
            # Contrastive 손실 (배치 내에서 쌍 구성)
            # 배치를 절반으로 나눠서 pair 구성
            batch_size = input_ids.size(0)
            if batch_size >= 4:
                half = batch_size // 2
                z_i, z_j = projection[:half], projection[half:2*half]
                labels_i, labels_j = labels[:half], labels[half:2*half]
                cl_loss = self.model.contrastive_loss(z_i, z_j, labels_i, labels_j)
            else:
                cl_loss = torch.tensor(0.0, device=self.device)
            
            # 총 손실
            loss = ce_loss + self.lambda_contrastive * cl_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_cl_loss += cl_loss.item() if isinstance(cl_loss, torch.Tensor) else cl_loss
        
        n_batches = len(self.train_loader)
        return {
            'total_loss': total_loss / n_batches,
            'ce_loss': total_ce_loss / n_batches,
            'contrastive_loss': total_cl_loss / n_batches
        }
    
    def evaluate(self):
        """
        검증 데이터 평가
        
        Returns:
            dict: 평가 결과 {'f1_macro', 'loss'}
        """
        self.model.eval()
        total_loss = 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                logits = self.model(input_ids, attention_mask)
                loss = self.ce_criterion(logits, labels)
                
                preds = (torch.sigmoid(logits) > 0.5).float()
                
                total_loss += loss.item()
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
        
        preds = torch.cat(all_preds).numpy()
        labels = torch.cat(all_labels).numpy()
        
        from sklearn.metrics import f1_score
        f1 = f1_score(labels, preds, average='macro', zero_division=0)
        
        return {
            'f1_macro': f1,
            'loss': total_loss / len(self.val_loader)
        }


# ============================================================
# 사용 예시
# ============================================================

def example_usage():
    """Contrastive Learning 사용 예시"""
    
    # 1. 모델 생성
    model = ContrastiveQualityModel(
        model_name='klue/roberta-base',
        num_labels=9,
        projection_dim=256,
        temperature=0.07
    )
    
    print(f"모델 파라미터: {sum(p.numel() for p in model.parameters()):,}")
    
    # 2. 토크나이저
    tokenizer = AutoTokenizer.from_pretrained('klue/roberta-base')
    
    # 3. 더미 입력
    texts = [
        "오늘 서울 날씨가 어때요?",
        "서울은 맑고 기온이 20도입니다."
    ]
    
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    
    # 4. 추론
    model.eval()
    with torch.no_grad():
        logits = model(inputs['input_ids'], inputs['attention_mask'])
        probs = torch.sigmoid(logits)
    
    print(f"예측 확률: {probs}")
    
    # 5. Projection 포함 추론
    with torch.no_grad():
        logits, projection = model(
            inputs['input_ids'], 
            inputs['attention_mask'],
            return_projection=True
        )
    
    print(f"Projection shape: {projection.shape}")


if __name__ == "__main__":
    example_usage()
