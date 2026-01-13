"""
데이터 전처리 모듈

Soft label 계산 및 데이터 전처리 함수를 제공합니다.
대화 맥락 동적 포함 기능을 지원합니다.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from tqdm import tqdm

# 평가 기준
CRITERIA = [
    'linguistic_acceptability',
    'consistency',
    'interestingness',
    'unbias',
    'harmlessness',
    'no_hallucination',
    'understandability',
    'sensibleness',
    'specificity'
]


def load_data(
    train_path: str,
    val_path: str,
    encoding: str = 'utf-8-sig'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    학습 및 검증 데이터 로드
    
    Args:
        train_path: 학습 데이터 경로
        val_path: 검증 데이터 경로
        encoding: 파일 인코딩
        
    Returns:
        (train_df, val_df) 튜플
    """
    train_df = pd.read_csv(train_path, encoding=encoding)
    val_df = pd.read_csv(val_path, encoding=encoding)
    
    print(f"Loaded {len(train_df):,} training samples")
    print(f"Loaded {len(val_df):,} validation samples")
    
    return train_df, val_df


def compute_soft_labels(
    row: pd.Series,
    criteria: List[str] = CRITERIA
) -> List[float]:
    """
    평가자 투표 비율을 soft label로 변환
    
    예시:
        yes_count=2, no_count=1 → soft_label = 0.67
        yes_count=3, no_count=0 → soft_label = 1.0
        yes_count=0, no_count=3 → soft_label = 0.0
    
    Args:
        row: 데이터프레임 행
        criteria: 평가 기준 리스트
        
    Returns:
        soft label 리스트
    """
    soft_labels = []
    
    for c in criteria:
        yes_col = f'{c}_yes_count'
        no_col = f'{c}_no_count'
        
        yes_count = row.get(yes_col, 0)
        no_count = row.get(no_col, 0)
        
        # NaN 처리
        yes_count = 0 if pd.isna(yes_count) else int(yes_count)
        no_count = 0 if pd.isna(no_count) else int(no_count)
        
        total = yes_count + no_count
        
        if total > 0:
            soft_label = yes_count / total
        else:
            soft_label = 0.5  # 정보 없으면 불확실
        
        soft_labels.append(soft_label)
    
    return soft_labels


def compute_hard_labels(
    row: pd.Series,
    criteria: List[str] = CRITERIA
) -> List[int]:
    """
    majority voting 결과를 hard label로 변환
    
    Args:
        row: 데이터프레임 행
        criteria: 평가 기준 리스트
        
    Returns:
        hard label 리스트 (0 또는 1)
    """
    hard_labels = []
    
    for c in criteria:
        majority_col = f'{c}_majority'
        label = row.get(majority_col, 0)
        
        # NaN 처리
        if pd.isna(label):
            label = 0
        
        hard_labels.append(int(label))
    
    return hard_labels


def compute_uncertainty(soft_labels: List[float]) -> List[float]:
    """
    각 라벨의 불확실성 계산
    
    soft_label이 0.5에 가까울수록 불확실성이 높음
    
    Args:
        soft_labels: soft label 리스트
        
    Returns:
        불확실성 리스트 (0~1, 1이 가장 불확실)
    """
    uncertainties = []
    for sl in soft_labels:
        # 0.5에서 멀어질수록 확실함 (0에 가까움)
        # 0.5에 가까울수록 불확실함 (1에 가까움)
        uncertainty = 1 - abs(sl - 0.5) * 2
        uncertainties.append(uncertainty)
    return uncertainties


def preprocess_data(
    df: pd.DataFrame,
    criteria: List[str] = CRITERIA,
    sep_token: str = '[SEP]',
    include_soft_labels: bool = True
) -> pd.DataFrame:
    """
    데이터 전처리
    
    - 결측치 처리
    - 입력 텍스트 생성
    - Hard/Soft label 계산
    - 불확실성 계산
    
    Args:
        df: 원본 데이터프레임
        criteria: 평가 기준 리스트
        sep_token: SEP 토큰 (모델에 따라 다름)
        include_soft_labels: soft label 포함 여부
        
    Returns:
        전처리된 데이터프레임
    """
    df = df.copy()
    
    # 결측치 처리
    df['human_question'] = df['human_question'].fillna('')
    df['bot_response'] = df['bot_response'].fillna('')
    
    # 입력 텍스트 생성
    df['input_text'] = df['human_question'] + f' {sep_token} ' + df['bot_response']
    
    # Hard labels (majority voting)
    hard_label_cols = [f'{c}_majority' for c in criteria]
    
    # 타겟 결측치가 있는 행 제거
    df = df.dropna(subset=hard_label_cols)
    
    # Hard labels를 리스트로 저장
    df['hard_labels'] = df.apply(
        lambda row: compute_hard_labels(row, criteria),
        axis=1
    )
    
    # Soft labels 계산
    if include_soft_labels:
        df['soft_labels'] = df.apply(
            lambda row: compute_soft_labels(row, criteria),
            axis=1
        )
        
        # 불확실성 계산
        df['uncertainty'] = df['soft_labels'].apply(compute_uncertainty)
        
        # 난이도 분류 (Curriculum Learning용)
        df['difficulty'] = df['soft_labels'].apply(classify_difficulty)
    
    return df


def classify_difficulty(soft_labels: List[float]) -> str:
    """
    샘플 난이도 분류 (Curriculum Learning용)
    
    만장일치에 가까울수록 쉬움
    
    Args:
        soft_labels: soft label 리스트
        
    Returns:
        'easy', 'medium', 'hard' 중 하나
    """
    # 모든 라벨이 0.9 이상이거나 0.1 이하면 쉬움 (만장일치에 가까움)
    if all(sl >= 0.9 or sl <= 0.1 for sl in soft_labels):
        return 'easy'
    # 하나라도 0.4~0.6 범위면 어려움 (의견이 갈림)
    elif any(0.4 <= sl <= 0.6 for sl in soft_labels):
        return 'hard'
    else:
        return 'medium'


def get_label_distribution(
    df: pd.DataFrame,
    criteria: List[str] = CRITERIA
) -> Dict[str, Dict[str, float]]:
    """
    레이블 분포 계산
    
    Args:
        df: 데이터프레임
        criteria: 평가 기준 리스트
        
    Returns:
        기준별 positive 비율 딕셔너리
    """
    distribution = {}
    
    for c in criteria:
        col = f'{c}_majority'
        if col in df.columns:
            pos_ratio = df[col].mean()
            neg_ratio = 1 - pos_ratio
            distribution[c] = {
                'positive_ratio': pos_ratio,
                'negative_ratio': neg_ratio,
                'count': len(df)
            }
    
    return distribution


def print_label_distribution(df: pd.DataFrame, criteria: List[str] = CRITERIA):
    """레이블 분포 출력"""
    print("\n" + "=" * 50)
    print("레이블 분포")
    print("=" * 50)
    
    distribution = get_label_distribution(df, criteria)
    
    for c, stats in distribution.items():
        print(f"{c}: {stats['positive_ratio']:.2%} positive")


def create_stratified_split(
    df: pd.DataFrame,
    test_size: float = 0.1,
    random_state: int = 42,
    criteria: List[str] = CRITERIA
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified split (레이블 분포 유지)
    
    Args:
        df: 데이터프레임
        test_size: 테스트 비율
        random_state: 랜덤 시드
        criteria: 평가 기준 리스트
        
    Returns:
        (train_df, test_df) 튜플
    """
    from sklearn.model_selection import train_test_split
    
    # 첫 번째 기준으로 stratify (multi-label이라 완벽한 stratify는 어려움)
    stratify_col = f'{criteria[0]}_majority'
    
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[stratify_col] if stratify_col in df.columns else None
    )
    
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def build_context_text(
    conversation_df: pd.DataFrame,
    current_idx: int,
    tokenizer: Any,
    max_length: int = 512,
    max_prev_turns: int = 7,
    sep_token: str = '[SEP]',
    turn_token: str = '[턴]',
    context_token: str = '[맥락]',
    current_token: str = '[현재]'
) -> Tuple[str, int]:
    """
    동적으로 이전 턴 맥락을 포함한 입력 텍스트 생성
    
    512 토큰 제한 내에서 최대한 많은 이전 턴을 포함합니다.
    가장 최근 턴부터 역순으로 추가하고, 토큰 초과 시 오래된 턴을 제외합니다.
    
    Args:
        conversation_df: 정렬된 대화 데이터프레임 (utterance_index 기준)
        current_idx: 현재 턴의 인덱스 (conversation_df 내 위치)
        tokenizer: 토크나이저
        max_length: 최대 토큰 길이
        max_prev_turns: 최대 포함할 이전 턴 수
        sep_token: SEP 토큰
        turn_token: 턴 구분 토큰
        context_token: 맥락 시작 토큰
        current_token: 현재 턴 시작 토큰
        
    Returns:
        (input_text, num_prev_turns) 튜플
    """
    current_row = conversation_df.iloc[current_idx]
    current_q = str(current_row['human_question'])
    current_a = str(current_row['bot_response'])
    
    # 현재 턴 텍스트 (항상 포함)
    current_text = f"{current_token} Q: {current_q} {sep_token} A: {current_a}"
    current_tokens = len(tokenizer.tokenize(current_text))
    
    # 이전 턴이 없거나 토큰이 이미 초과하면 현재만 반환
    if current_idx == 0 or current_tokens >= max_length:
        simple_text = f"Q: {current_q} {sep_token} A: {current_a}"
        return simple_text, 0
    
    # 이전 턴들 수집 (최신순으로)
    prev_turns = []
    # 특수 토큰([CLS], [SEP], [PAD] 등) 및 여유분 확보 (100 토큰 - 안전 마진)
    available_tokens = max_length - current_tokens - 100
    
    for i in range(current_idx - 1, max(current_idx - max_prev_turns - 1, -1), -1):
        prev_row = conversation_df.iloc[i]
        prev_q = str(prev_row['human_question'])
        prev_a = str(prev_row['bot_response'])
        prev_text = f"Q: {prev_q} A: {prev_a}"
        prev_tokens = len(tokenizer.tokenize(prev_text)) + 3  # 턴 토큰 추가
        
        if prev_tokens <= available_tokens:
            prev_turns.insert(0, prev_text)  # 앞에 추가 (시간 순서 유지)
            available_tokens -= prev_tokens
        else:
            break  # 토큰 초과 시 중단
    
    # 최종 텍스트 구성
    if prev_turns:
        context_text = f" {turn_token} ".join(prev_turns)
        final_text = f"{context_token} {context_text} {current_text}"
    else:
        final_text = f"Q: {current_q} {sep_token} A: {current_a}"
    
    return final_text, len(prev_turns)


def preprocess_data_with_context(
    df: pd.DataFrame,
    tokenizer: Any,
    criteria: List[str] = CRITERIA,
    max_length: int = 512,
    max_prev_turns: int = 7,
    sep_token: str = '[SEP]',
    include_soft_labels: bool = True,
    show_progress: bool = True
) -> pd.DataFrame:
    """
    대화 맥락을 동적으로 포함하는 데이터 전처리
    
    각 샘플에 대해 512 토큰 제한 내에서 최대한 많은 이전 턴을 포함합니다.
    
    Args:
        df: 원본 데이터프레임
        tokenizer: 토크나이저 (토큰 길이 계산용)
        criteria: 평가 기준 리스트
        max_length: 최대 토큰 길이
        max_prev_turns: 최대 포함할 이전 턴 수
        sep_token: SEP 토큰
        include_soft_labels: soft label 포함 여부
        show_progress: 진행 상황 표시 여부
        
    Returns:
        전처리된 데이터프레임
    """
    df = df.copy()
    
    # 결측치 처리
    df['human_question'] = df['human_question'].fillna('')
    df['bot_response'] = df['bot_response'].fillna('')
    
    # Hard labels (majority voting) 컬럼 확인
    hard_label_cols = [f'{c}_majority' for c in criteria]
    df = df.dropna(subset=hard_label_cols)
    
    # conversation_id별로 그룹화하여 처리
    input_texts = []
    num_prev_turns_list = []
    
    # 대화별로 정렬된 인덱스 매핑 생성
    conversation_groups = df.groupby('conversation_id')
    
    # 진행 상황 표시
    iterator = tqdm(df.iterrows(), total=len(df), desc="Building context") if show_progress else df.iterrows()
    
    for idx, row in iterator:
        conv_id = row['conversation_id']
        conv_df = conversation_groups.get_group(conv_id).sort_values('utterance_index').reset_index(drop=True)
        
        # 현재 행의 conversation 내 위치 찾기
        current_utterance_idx = row['utterance_index']
        current_pos = conv_df[conv_df['utterance_index'] == current_utterance_idx].index[0]
        
        # 맥락 포함 텍스트 생성
        input_text, num_prev = build_context_text(
            conversation_df=conv_df,
            current_idx=current_pos,
            tokenizer=tokenizer,
            max_length=max_length,
            max_prev_turns=max_prev_turns,
            sep_token=sep_token
        )
        
        input_texts.append(input_text)
        num_prev_turns_list.append(num_prev)
    
    df['input_text'] = input_texts
    df['num_prev_turns'] = num_prev_turns_list
    
    # Hard labels를 리스트로 저장
    df['hard_labels'] = df.apply(
        lambda row: compute_hard_labels(row, criteria),
        axis=1
    )
    
    # Soft labels 계산
    if include_soft_labels:
        df['soft_labels'] = df.apply(
            lambda row: compute_soft_labels(row, criteria),
            axis=1
        )
        
        # 불확실성 계산
        df['uncertainty'] = df['soft_labels'].apply(compute_uncertainty)
        
        # 난이도 분류 (Curriculum Learning용)
        df['difficulty'] = df['soft_labels'].apply(classify_difficulty)
    
    # 통계 출력
    if show_progress:
        print(f"\n=== 맥락 포함 통계 ===")
        print(f"총 샘플 수: {len(df):,}")
        print(f"평균 이전 턴 수: {np.mean(num_prev_turns_list):.2f}")
        print(f"최대 이전 턴 수: {max(num_prev_turns_list)}")
        turns_dist = pd.Series(num_prev_turns_list).value_counts().sort_index()
        print(f"이전 턴 수 분포:")
        for n, count in turns_dist.items():
            print(f"  {n}턴: {count:,} ({count/len(df)*100:.1f}%)")
    
    return df


def preprocess_for_crossencoder(
    df: pd.DataFrame,
    tokenizer: Any,
    criteria: List[str] = CRITERIA,
    max_length: int = 512,
    max_prev_turns: int = 7,
    include_soft_labels: bool = True,
    show_progress: bool = True
) -> pd.DataFrame:
    """
    Cross-Encoder용 전처리 (맥락+질문, 응답을 pair로 구성)
    
    입력 형식: [맥락 + 질문] [SEP] [응답]
    환각 탐지에 특화된 형식입니다.
    
    Args:
        df: 원본 데이터프레임
        tokenizer: 토크나이저
        criteria: 평가 기준 리스트
        max_length: 최대 토큰 길이
        max_prev_turns: 최대 포함할 이전 턴 수
        include_soft_labels: soft label 포함 여부
        show_progress: 진행 상황 표시 여부
        
    Returns:
        전처리된 데이터프레임 (text_a, text_b 컬럼 포함)
    """
    df = df.copy()
    
    # 결측치 처리
    df['human_question'] = df['human_question'].fillna('')
    df['bot_response'] = df['bot_response'].fillna('')
    
    # Hard labels 컬럼 확인
    hard_label_cols = [f'{c}_majority' for c in criteria]
    df = df.dropna(subset=hard_label_cols)
    
    text_a_list = []  # 맥락 + 질문
    text_b_list = []  # 응답
    num_prev_turns_list = []
    
    conversation_groups = df.groupby('conversation_id')
    iterator = tqdm(df.iterrows(), total=len(df), desc="Building cross-encoder pairs") if show_progress else df.iterrows()
    
    for idx, row in iterator:
        conv_id = row['conversation_id']
        conv_df = conversation_groups.get_group(conv_id).sort_values('utterance_index').reset_index(drop=True)
        
        current_utterance_idx = row['utterance_index']
        current_pos = conv_df[conv_df['utterance_index'] == current_utterance_idx].index[0]
        
        current_q = str(row['human_question'])
        current_a = str(row['bot_response'])
        
        # text_b는 항상 현재 응답
        text_b = current_a
        text_b_tokens = len(tokenizer.tokenize(text_b))
        
        # text_a에 맥락 + 질문 포함
        available_tokens = max_length - text_b_tokens - 10  # SEP 토큰 등 여유
        
        # 현재 질문
        current_q_text = f"[질문] {current_q}"
        current_q_tokens = len(tokenizer.tokenize(current_q_text))
        
        if current_q_tokens >= available_tokens:
            # 질문만으로도 초과하면 질문만 포함
            text_a = current_q_text
            num_prev = 0
        else:
            # 이전 턴 추가
            available_tokens -= current_q_tokens
            prev_turns = []
            
            for i in range(current_pos - 1, max(current_pos - max_prev_turns - 1, -1), -1):
                prev_row = conv_df.iloc[i]
                prev_q = str(prev_row['human_question'])
                prev_a = str(prev_row['bot_response'])
                prev_text = f"Q: {prev_q} A: {prev_a}"
                prev_tokens = len(tokenizer.tokenize(prev_text)) + 3
                
                if prev_tokens <= available_tokens:
                    prev_turns.insert(0, prev_text)
                    available_tokens -= prev_tokens
                else:
                    break
            
            if prev_turns:
                context_text = " [턴] ".join(prev_turns)
                text_a = f"[맥락] {context_text} {current_q_text}"
            else:
                text_a = current_q_text
            
            num_prev = len(prev_turns)
        
        text_a_list.append(text_a)
        text_b_list.append(text_b)
        num_prev_turns_list.append(num_prev)
    
    df['text_a'] = text_a_list  # 맥락 + 질문
    df['text_b'] = text_b_list  # 응답
    df['input_text'] = df['text_a'] + ' [SEP] ' + df['text_b']  # 호환성
    df['num_prev_turns'] = num_prev_turns_list
    
    # Labels
    df['hard_labels'] = df.apply(
        lambda row: compute_hard_labels(row, criteria),
        axis=1
    )
    
    if include_soft_labels:
        df['soft_labels'] = df.apply(
            lambda row: compute_soft_labels(row, criteria),
            axis=1
        )
        df['uncertainty'] = df['soft_labels'].apply(compute_uncertainty)
        df['difficulty'] = df['soft_labels'].apply(classify_difficulty)
    
    if show_progress:
        print(f"\n=== Cross-Encoder 맥락 포함 통계 ===")
        print(f"총 샘플 수: {len(df):,}")
        print(f"평균 이전 턴 수: {np.mean(num_prev_turns_list):.2f}")
    
    return df
