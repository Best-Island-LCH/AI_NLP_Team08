"""
Cross-Encoder for AI Quality Evaluation

Context-Response 상호작용 모델
- Dual Encoding
- Cross-Attention
- Attention Pooling
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class CrossEncoderModel(nn.Module):
    """
    Cross-Encoder: Context-Response 상호작용 모델
    
    Context와 Response를 별도로 인코딩한 후
    Cross-Attention으로 상호작용 모델링
    """
    
    def __init__(
        self,
        model_name: str = 'klue/roberta-base',
        num_labels: int = 9,
        num_cross_layers: int = 2,
        dropout_prob: float = 0.1
    ):
        """
        Args:
            model_name: HuggingFace 모델 ID
            num_labels: 출력 라벨 수
            num_cross_layers: Cross-Attention 레이어 수
            dropout_prob: 드롭아웃 확률
        """
        super().__init__()
        
        self.num_labels = num_labels
        
        # Shared BERT Encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        
        # Cross-Attention Layers
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(
                hidden_size=self.hidden_size,
                num_heads=8,
                dropout_prob=dropout_prob
            )
            for _ in range(num_cross_layers)
        ])
        
        # Pooling
        self.pooler = AttentionPooling(self.hidden_size)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(self.hidden_size * 2, 512),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(512, num_labels)
        )
    
    def forward(
        self,
        context_input_ids: torch.Tensor,
        context_attention_mask: torch.Tensor,
        response_input_ids: torch.Tensor,
        response_attention_mask: torch.Tensor
    ):
        """
        순전파
        
        Args:
            context_input_ids: [batch, seq_len_c]
            context_attention_mask: [batch, seq_len_c]
            response_input_ids: [batch, seq_len_r]
            response_attention_mask: [batch, seq_len_r]
        
        Returns:
            logits: [batch, num_labels]
            cross_attention_weights: 디버깅용 어텐션 가중치 리스트
        """
        # Context 인코딩
        context_outputs = self.encoder(
            input_ids=context_input_ids,
            attention_mask=context_attention_mask
        )
        context_hidden = context_outputs.last_hidden_state  # [batch, seq_c, hidden]
        
        # Response 인코딩
        response_outputs = self.encoder(
            input_ids=response_input_ids,
            attention_mask=response_attention_mask
        )
        response_hidden = response_outputs.last_hidden_state  # [batch, seq_r, hidden]
        
        # Cross-Attention (Response가 Context를 참조)
        cross_attended = response_hidden
        attention_weights_list = []
        
        for cross_layer in self.cross_attention_layers:
            cross_attended, attn_weights = cross_layer(
                query=cross_attended,
                key_value=context_hidden,
                key_padding_mask=~context_attention_mask.bool()
            )
            attention_weights_list.append(attn_weights)
        
        # Pooling
        response_pooled = self.pooler(
            cross_attended, response_attention_mask
        )  # [batch, hidden]
        
        context_pooled = context_outputs.last_hidden_state[:, 0, :]  # [CLS]
        
        # 결합
        combined = torch.cat([context_pooled, response_pooled], dim=-1)
        
        # 분류
        logits = self.classifier(combined)
        
        return logits, attention_weights_list


class CrossAttentionLayer(nn.Module):
    """
    Cross-Attention 레이어
    
    Response가 Context를 참조하는 어텐션
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        dropout_prob: float = 0.1
    ):
        super().__init__()
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout_prob,
            batch_first=True
        )
        
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout_prob)
        )
    
    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: torch.Tensor = None
    ):
        """
        Args:
            query: [batch, seq_q, hidden] (Response)
            key_value: [batch, seq_kv, hidden] (Context)
            key_padding_mask: [batch, seq_kv] (True = 패딩)
        
        Returns:
            output: [batch, seq_q, hidden]
            attn_weights: [batch, seq_q, seq_kv]
        """
        # Cross-Attention
        attn_output, attn_weights = self.cross_attention(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=key_padding_mask
        )
        
        # Residual + LayerNorm
        query = self.layer_norm1(query + attn_output)
        
        # Feed-Forward
        ff_output = self.feed_forward(query)
        output = self.layer_norm2(query + ff_output)
        
        return output, attn_weights


class AttentionPooling(nn.Module):
    """
    어텐션 기반 풀링
    
    시퀀스를 단일 벡터로 변환
    """
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, hidden]
            attention_mask: [batch, seq_len]
        
        Returns:
            pooled: [batch, hidden]
        """
        # 어텐션 스코어
        scores = self.attention(hidden_states).squeeze(-1)  # [batch, seq_len]
        
        # 마스킹
        scores = scores.masked_fill(~attention_mask.bool(), float('-inf'))
        
        # 소프트맥스
        weights = torch.softmax(scores, dim=-1)  # [batch, seq_len]
        
        # 가중 평균
        pooled = torch.bmm(
            weights.unsqueeze(1), hidden_states
        ).squeeze(1)  # [batch, hidden]
        
        return pooled


class CrossEncoderDataset(torch.utils.data.Dataset):
    """
    Cross-Encoder를 위한 데이터셋
    Context와 Response를 분리하여 제공
    """
    
    CRITERIA = [
        'linguistic_acceptability', 'consistency', 'interestingness',
        'unbias', 'harmlessness', 'no_hallucination',
        'understandability', 'sensibleness', 'specificity'
    ]
    
    def __init__(
        self,
        samples: list,
        tokenizer,
        max_context_length: int = 256,
        max_response_length: int = 128
    ):
        """
        Args:
            samples: 샘플 리스트
            tokenizer: 토크나이저
            max_context_length: Context 최대 길이
            max_response_length: Response 최대 길이
        """
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_context_length = max_context_length
        self.max_response_length = max_response_length
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        
        # Context (이전 대화)
        context_text = sample.get('human_question', '')
        if 'context' in sample:
            context_text = sample['context'] + ' ' + context_text
        
        # Response (평가 대상)
        response_text = sample.get('bot_response', '')
        
        # 토크나이징
        context_encoding = self.tokenizer(
            context_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_context_length,
            return_tensors='pt'
        )
        
        response_encoding = self.tokenizer(
            response_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_response_length,
            return_tensors='pt'
        )
        
        # 라벨
        labels = []
        for c in self.CRITERIA:
            if f'{c}_majority' in sample:
                labels.append(float(sample[f'{c}_majority']))
            elif c in sample:
                labels.append(float(sample[c]))
            else:
                labels.append(0.0)
        
        return {
            'context_input_ids': context_encoding['input_ids'].squeeze(),
            'context_attention_mask': context_encoding['attention_mask'].squeeze(),
            'response_input_ids': response_encoding['input_ids'].squeeze(),
            'response_attention_mask': response_encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(labels, dtype=torch.float)
        }


# ============================================================
# 사용 예시
# ============================================================

def example_usage():
    """Cross-Encoder 사용 예시"""
    
    print("Cross-Encoder Model 예시")
    print("=" * 50)
    
    tokenizer = AutoTokenizer.from_pretrained('klue/roberta-base')
    
    model = CrossEncoderModel(
        model_name='klue/roberta-base',
        num_labels=9,
        num_cross_layers=2
    )
    
    print(f"모델 파라미터: {sum(p.numel() for p in model.parameters()):,}")
    
    # 예시 입력
    context = "서울의 오늘 날씨와 내일 날씨 알려줘"
    response = "오늘 서울은 맑고 기온 20도입니다. 내일은 비가 올 예정입니다."
    
    # 토크나이징
    context_enc = tokenizer(
        context, 
        return_tensors='pt', 
        padding='max_length', 
        max_length=128,
        truncation=True
    )
    response_enc = tokenizer(
        response, 
        return_tensors='pt', 
        padding='max_length', 
        max_length=128,
        truncation=True
    )
    
    # 추론
    model.eval()
    with torch.no_grad():
        logits, attn_weights = model(
            context_enc['input_ids'],
            context_enc['attention_mask'],
            response_enc['input_ids'],
            response_enc['attention_mask']
        )
        probs = torch.sigmoid(logits)
    
    print(f"\n입력:")
    print(f"  Context: {context}")
    print(f"  Response: {response}")
    print(f"\n예측 확률: {probs.numpy().round(3)}")
    print(f"Cross-attention layers: {len(attn_weights)}")


if __name__ == "__main__":
    example_usage()
