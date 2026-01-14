"""
Contrastive Learning for AI Quality Evaluation

Supervised Contrastive Learning을 통해 더 좋은 임베딩을 학습합니다.
같은 품질의 응답은 가깝게, 다른 품질의 응답은 멀리 배치합니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from typing import Dict, Optional, Tuple


class ContrastiveQualityModel(nn.Module):
    """
    Supervised Contrastive Learning 모델
    
    - Projection Head로 대조 학습용 임베딩 생성
    - Classification Head로 최종 예측
    - BCE Loss + Contrastive Loss 조합
    """
    
    def __init__(
        self,
        model_name: str = 'klue/roberta-base',
        num_labels: int = 9,
        projection_dim: int = 256,
        hidden_dropout_prob: float = 0.1,
        temperature: float = 0.07
    ):
        """
        Args:
            model_name: HuggingFace 모델 ID
            num_labels: 출력 라벨 수
            projection_dim: Projection Head 출력 차원
            hidden_dropout_prob: 드롭아웃 확률
            temperature: Contrastive Loss 온도 파라미터
        """
        super().__init__()
        
        self.num_labels = num_labels
        self.temperature = temperature
        
        # BERT Encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        
        # Projection Head (대조 학습용)
        self.projection_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(hidden_dropout_prob),
            nn.Linear(self.hidden_size, projection_dim)
        )
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(hidden_dropout_prob),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(hidden_dropout_prob),
            nn.Linear(self.hidden_size // 2, num_labels)
        )
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        순전파
        
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            labels: [batch, num_labels] (선택)
            
        Returns:
            logits: 분류 로짓
            embeddings: 정규화된 대조 학습 임베딩
            loss: 손실 (labels 제공 시)
        """
        # BERT 인코딩
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch, hidden]
        
        # Projection (대조 학습용)
        projected = self.projection_head(cls_output)  # [batch, proj_dim]
        embeddings = F.normalize(projected, p=2, dim=1)  # L2 정규화
        
        # Classification
        logits = self.classifier(cls_output)  # [batch, num_labels]
        
        # 손실 계산
        loss = None
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        
        return {
            'logits': logits,
            'embeddings': embeddings,
            'loss': loss
        }
    
    def get_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """임베딩만 반환 (대조 학습 배치 구성용)"""
        with torch.no_grad():
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            cls_output = outputs.last_hidden_state[:, 0, :]
            projected = self.projection_head(cls_output)
            embeddings = F.normalize(projected, p=2, dim=1)
        return embeddings


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss
    
    같은 라벨을 가진 샘플들은 가깝게, 다른 라벨은 멀게 학습합니다.
    Multi-label 설정에서는 라벨 유사도를 기반으로 합니다.
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        base_temperature: float = 0.07
    ):
        """
        Args:
            temperature: 소프트맥스 온도
            base_temperature: 기본 온도 (스케일링용)
        """
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            embeddings: [batch, proj_dim] 정규화된 임베딩
            labels: [batch, num_labels] 멀티라벨
            
        Returns:
            contrastive loss
        """
        device = embeddings.device
        batch_size = embeddings.shape[0]
        
        if batch_size < 2:
            return torch.tensor(0.0, device=device)
        
        # 라벨 유사도 계산 (Jaccard similarity)
        # 라벨이 완전히 같으면 1, 전혀 다르면 0
        label_similarity = self._compute_label_similarity(labels)
        
        # 유사도 임계값 (0.5 이상이면 positive pair로 취급)
        positive_mask = (label_similarity > 0.5).float()
        
        # 자기 자신 제외
        eye_mask = 1 - torch.eye(batch_size, device=device)
        positive_mask = positive_mask * eye_mask
        
        # 코사인 유사도 (이미 L2 정규화되어 있음)
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # 안정성을 위해 최대값 빼기
        sim_matrix_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        sim_matrix = sim_matrix - sim_matrix_max.detach()
        
        # exp 계산
        exp_sim = torch.exp(sim_matrix) * eye_mask
        
        # Contrastive loss
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        
        # Positive pair에 대해서만 loss 계산
        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / (positive_mask.sum(dim=1) + 1e-8)
        
        # Temperature scaling
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        
        return loss
    
    def _compute_label_similarity(self, labels: torch.Tensor) -> torch.Tensor:
        """
        멀티라벨 간 Jaccard 유사도 계산
        
        J(A, B) = |A ∩ B| / |A ∪ B|
        """
        batch_size = labels.shape[0]
        labels = labels.float()
        
        # 교집합: AND 연산
        intersection = torch.matmul(labels, labels.T)  # [batch, batch]
        
        # 합집합: |A| + |B| - |A ∩ B|
        label_sums = labels.sum(dim=1, keepdim=True)  # [batch, 1]
        union = label_sums + label_sums.T - intersection
        
        # Jaccard similarity
        similarity = intersection / (union + 1e-8)
        
        return similarity


class ContrastiveLossWithBCE(nn.Module):
    """
    BCE Loss + Contrastive Loss 조합
    
    lambda_contrastive로 두 손실의 비율을 조절합니다.
    """
    
    def __init__(
        self,
        lambda_contrastive: float = 0.1,
        temperature: float = 0.07
    ):
        """
        Args:
            lambda_contrastive: Contrastive Loss 가중치
            temperature: Contrastive Loss 온도
        """
        super().__init__()
        self.lambda_contrastive = lambda_contrastive
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.contrastive_loss = SupervisedContrastiveLoss(temperature=temperature)
    
    def forward(
        self,
        logits: torch.Tensor,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            logits: 분류 로짓 [batch, num_labels]
            embeddings: 대조 학습 임베딩 [batch, proj_dim]
            labels: 라벨 [batch, num_labels]
            
        Returns:
            total_loss, loss_dict
        """
        # BCE Loss
        bce = self.bce_loss(logits, labels.float())
        
        # Contrastive Loss
        contrastive = self.contrastive_loss(embeddings, labels)
        
        # 총 손실
        total_loss = bce + self.lambda_contrastive * contrastive
        
        loss_dict = {
            'bce_loss': bce.item(),
            'contrastive_loss': contrastive.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_dict


def get_contrastive_model(
    model_name: str = 'klue/roberta-base',
    projection_dim: int = 256,
    temperature: float = 0.07,
    **kwargs
) -> ContrastiveQualityModel:
    """
    Contrastive Learning 모델 생성 헬퍼
    
    Args:
        model_name: HuggingFace 모델 ID
        projection_dim: Projection Head 출력 차원
        temperature: Contrastive Loss 온도
        **kwargs: 추가 인자
        
    Returns:
        ContrastiveQualityModel 인스턴스
    """
    return ContrastiveQualityModel(
        model_name=model_name,
        projection_dim=projection_dim,
        temperature=temperature,
        **kwargs
    )
