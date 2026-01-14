"""
토큰 분포 분석 스크립트 (Phase 0)

각 모델의 토크나이저로 데이터셋의 토큰 길이 분포를 분석하여
적정 max_length를 결정합니다.
"""

import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
import json
from pathlib import Path
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# 분석할 모델들
MODELS = {
    'klue/roberta-base': 'klue/roberta-base',
    'monologg/koelectra-base-v3-discriminator': 'monologg/koelectra-base-v3-discriminator',
    'monologg/kobert': 'monologg/kobert',
    'klue/roberta-large': 'klue/roberta-large',
}

# 평가 기준
CRITERIA = [
    'linguistic_acceptability', 'consistency', 'interestingness',
    'unbias', 'harmlessness', 'no_hallucination',
    'understandability', 'sensibleness', 'specificity'
]


def load_data(data_dir: str):
    """데이터 로드"""
    train_path = Path(data_dir) / 'train' / 'training_all_aggregated.csv'
    val_path = Path(data_dir) / 'val' / 'validation_all_aggregated.csv'
    
    train_df = pd.read_csv(train_path, encoding='utf-8-sig')
    val_df = pd.read_csv(val_path, encoding='utf-8-sig')
    
    print(f"Train samples: {len(train_df):,}")
    print(f"Val samples: {len(val_df):,}")
    
    return train_df, val_df


def preprocess_text(df: pd.DataFrame) -> pd.DataFrame:
    """입력 텍스트 생성"""
    df = df.copy()
    df['human_question'] = df['human_question'].fillna('')
    df['bot_response'] = df['bot_response'].fillna('')
    df['input_text'] = df['human_question'] + ' [SEP] ' + df['bot_response']
    return df


def analyze_token_distribution(df: pd.DataFrame, tokenizer, text_column: str = 'input_text', 
                                sample_size: int = None):
    """토큰 길이 분포 분석"""
    if sample_size and len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
    else:
        df_sample = df
    
    lengths = []
    for text in tqdm(df_sample[text_column], desc="Tokenizing"):
        if pd.isna(text) or text == '':
            lengths.append(0)
            continue
        try:
            tokens = tokenizer(str(text), truncation=False, add_special_tokens=True)
            lengths.append(len(tokens['input_ids']))
        except Exception as e:
            print(f"Error tokenizing: {e}")
            lengths.append(0)
    
    lengths = np.array(lengths)
    
    stats = {
        'count': len(lengths),
        'mean': float(np.mean(lengths)),
        'std': float(np.std(lengths)),
        'min': int(np.min(lengths)),
        'p25': int(np.percentile(lengths, 25)),
        'p50': int(np.percentile(lengths, 50)),
        'p75': int(np.percentile(lengths, 75)),
        'p90': int(np.percentile(lengths, 90)),
        'p95': int(np.percentile(lengths, 95)),
        'p99': int(np.percentile(lengths, 99)),
        'max': int(np.max(lengths)),
    }
    
    # 권장 max_length 계산
    stats['recommended_max_lengths'] = {
        'conservative': stats['p90'],  # 90% 커버리지
        'balanced': stats['p95'],      # 95% 커버리지
        'aggressive': stats['p99'],    # 99% 커버리지
    }
    
    return stats, lengths


def plot_distribution(lengths_dict: dict, output_path: str):
    """토큰 길이 분포 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (model_name, lengths) in enumerate(lengths_dict.items()):
        ax = axes[idx]
        ax.hist(lengths, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(np.percentile(lengths, 90), color='r', linestyle='--', label=f'P90: {int(np.percentile(lengths, 90))}')
        ax.axvline(np.percentile(lengths, 95), color='g', linestyle='--', label=f'P95: {int(np.percentile(lengths, 95))}')
        ax.axvline(np.percentile(lengths, 99), color='b', linestyle='--', label=f'P99: {int(np.percentile(lengths, 99))}')
        ax.set_title(f'{model_name}', fontsize=12)
        ax.set_xlabel('Token Length')
        ax.set_ylabel('Frequency')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Distribution plot saved to: {output_path}")


def main():
    # 데이터 경로
    data_dir = Path(__file__).parent.parent.parent / 'data'
    output_dir = Path(__file__).parent.parent / 'config'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("토큰 분포 분석 (Phase 0)")
    print("=" * 60)
    
    # 데이터 로드
    train_df, val_df = load_data(data_dir)
    
    # 텍스트 전처리
    train_df = preprocess_text(train_df)
    val_df = preprocess_text(val_df)
    
    # 전체 데이터 합치기
    all_df = pd.concat([train_df, val_df], ignore_index=True)
    print(f"\nTotal samples: {len(all_df):,}")
    
    # 샘플 크기 (전체 데이터가 크면 샘플링)
    sample_size = min(50000, len(all_df))
    print(f"Analyzing {sample_size:,} samples...")
    
    results = {}
    lengths_dict = {}
    
    for model_key, model_name in MODELS.items():
        print(f"\n{'=' * 60}")
        print(f"Analyzing: {model_name}")
        print("=" * 60)
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            stats, lengths = analyze_token_distribution(all_df, tokenizer, sample_size=sample_size)
            results[model_key] = stats
            lengths_dict[model_key] = lengths
            
            print(f"\n📊 Statistics for {model_name}:")
            print(f"  Mean: {stats['mean']:.1f} ± {stats['std']:.1f}")
            print(f"  P50 (Median): {stats['p50']}")
            print(f"  P75: {stats['p75']}")
            print(f"  P90: {stats['p90']}")
            print(f"  P95: {stats['p95']}")
            print(f"  P99: {stats['p99']}")
            print(f"  Max: {stats['max']}")
            print(f"\n  📌 Recommended max_length:")
            print(f"    Conservative (90%): {stats['recommended_max_lengths']['conservative']}")
            print(f"    Balanced (95%): {stats['recommended_max_lengths']['balanced']}")
            print(f"    Aggressive (99%): {stats['recommended_max_lengths']['aggressive']}")
            
        except Exception as e:
            print(f"  ❌ Error loading {model_name}: {e}")
            results[model_key] = {'error': str(e)}
    
    # 결과 저장
    output_file = output_dir / 'token_analysis_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Results saved to: {output_file}")
    
    # 분포 시각화
    if lengths_dict:
        plot_path = output_dir / 'token_distribution.png'
        plot_distribution(lengths_dict, str(plot_path))
    
    # 권장 설정 출력
    print("\n" + "=" * 60)
    print("📋 권장 max_length 설정 (Sweep용)")
    print("=" * 60)
    
    if 'klue/roberta-base' in results and 'error' not in results['klue/roberta-base']:
        base_stats = results['klue/roberta-base']
        print(f"""
sweep_config.yaml에 사용할 값:
  max_length:
    values: [{base_stats['p90']}, {base_stats['p95']}, {base_stats['p99']}]
""")
    
    return results


if __name__ == '__main__':
    results = main()
