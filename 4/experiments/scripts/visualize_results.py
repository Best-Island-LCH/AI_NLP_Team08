#!/usr/bin/env python
"""
실험 결과 시각화 스크립트

사용법:
    python scripts/visualize_results.py --phase phase1
    python scripts/visualize_results.py --all
    python scripts/visualize_results.py --results_dir outputs/results
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정
plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

# 평가 기준
CRITERIA = [
    'linguistic_acceptability', 'consistency', 'interestingness',
    'unbias', 'harmlessness', 'no_hallucination',
    'understandability', 'sensibleness', 'specificity'
]

# 모델 표시명
MODEL_DISPLAY_NAMES = {
    'klue/bert-base': 'BERT',
    'klue/roberta-base': 'RoBERTa',
    'monologg/koelectra-base-v3-discriminator': 'ELECTRA',
    'monologg/distilkobert': 'DistilBERT',
    'team-lucid/deberta-v3-base-korean': 'DeBERTa'
}

# Loss 표시명
LOSS_DISPLAY_NAMES = {
    'bce': 'BCE',
    'soft_bce': 'Soft BCE',
    'focal': 'Focal',
    'asl': 'ASL',
    'criterion_weighted': 'Weighted'
}

# 색상 팔레트
COLORS = {
    'bert': '#1f77b4',
    'roberta': '#ff7f0e',
    'electra': '#2ca02c',
    'distilbert': '#d62728',
    'deberta': '#9467bd'
}


def load_results(results_path: Path) -> Dict[str, Any]:
    """결과 JSON 파일 로드"""
    if not results_path.exists():
        return {}
    
    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_results(results: Dict[str, Any], results_path: Path):
    """결과 JSON 파일 저장"""
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def extract_model_name(model_id: str) -> str:
    """모델 ID에서 표시명 추출"""
    return MODEL_DISPLAY_NAMES.get(model_id, model_id.split('/')[-1])


def plot_architecture_comparison(results: Dict[str, Any], output_path: Path):
    """
    Phase 1: 아키텍처 비교 차트
    - 5개 모델 × 맥락 유/무 비교
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 데이터 준비
    models = []
    macro_f1_noctx = []
    macro_f1_ctx = []
    no_hall_f1_noctx = []
    no_hall_f1_ctx = []
    
    for exp_name, exp_data in results.items():
        if 'noctx' in exp_name:
            model_key = exp_name.replace('p1-', '').replace('-noctx', '')
            model_name = MODEL_DISPLAY_NAMES.get(exp_data.get('model', ''), model_key.upper())
            models.append(model_name)
            macro_f1_noctx.append(exp_data.get('macro_f1', 0) * 100)
            no_hall_f1_noctx.append(exp_data.get('no_hallucination_f1', 0) * 100)
        elif 'ctx' in exp_name:
            macro_f1_ctx.append(exp_data.get('macro_f1', 0) * 100)
            no_hall_f1_ctx.append(exp_data.get('no_hallucination_f1', 0) * 100)
    
    if not models:
        print("No Phase 1 results found")
        return
    
    x = np.arange(len(models))
    width = 0.35
    
    # Macro F1 비교
    ax1 = axes[0]
    bars1 = ax1.bar(x - width/2, macro_f1_noctx, width, label='No Context', color='#3498db')
    bars2 = ax1.bar(x + width/2, macro_f1_ctx, width, label='With Context', color='#e74c3c')
    
    ax1.set_ylabel('Macro F1 (%)', fontsize=12)
    ax1.set_title('Architecture Comparison - Macro F1', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend()
    ax1.set_ylim([min(macro_f1_noctx + macro_f1_ctx) - 2, 100])
    
    # 값 표시
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    # no_hallucination F1 비교
    ax2 = axes[1]
    bars3 = ax2.bar(x - width/2, no_hall_f1_noctx, width, label='No Context', color='#3498db')
    bars4 = ax2.bar(x + width/2, no_hall_f1_ctx, width, label='With Context', color='#e74c3c')
    
    ax2.set_ylabel('no_hallucination F1 (%)', fontsize=12)
    ax2.set_title('Architecture Comparison - no_hallucination F1', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.legend()
    ax2.set_ylim([min(no_hall_f1_noctx + no_hall_f1_ctx) - 2, 100])
    
    for bar in bars3 + bars4:
        height = bar.get_height()
        ax2.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_context_effect(results: Dict[str, Any], output_path: Path):
    """
    맥락 포함 효과 시각화
    - 각 모델별 맥락 유/무 차이
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = []
    improvements = []
    
    for exp_name, exp_data in results.items():
        if 'noctx' in exp_name:
            model_key = exp_name.replace('p1-', '').replace('-noctx', '')
            ctx_name = exp_name.replace('-noctx', '-ctx')
            
            if ctx_name in results:
                model_name = MODEL_DISPLAY_NAMES.get(exp_data.get('model', ''), model_key.upper())
                models.append(model_name)
                
                noctx_f1 = exp_data.get('macro_f1', 0) * 100
                ctx_f1 = results[ctx_name].get('macro_f1', 0) * 100
                improvements.append(ctx_f1 - noctx_f1)
    
    if not models:
        print("No context comparison data found")
        return
    
    colors = ['#27ae60' if imp > 0 else '#e74c3c' for imp in improvements]
    bars = ax.barh(models, improvements, color=colors)
    
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Macro F1 Improvement (%)', fontsize=12)
    ax.set_title('Context Effect on Each Architecture', fontsize=14, fontweight='bold')
    
    for i, (bar, imp) in enumerate(zip(bars, improvements)):
        ax.annotate(f'{imp:+.2f}%',
                   xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                   xytext=(5 if imp > 0 else -5, 0),
                   textcoords="offset points",
                   ha='left' if imp > 0 else 'right',
                   va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_loss_comparison(results: Dict[str, Any], output_path: Path):
    """
    Phase 2: Loss 함수 비교 차트
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 모델별로 그룹화
    model_results = {}
    
    for exp_name, exp_data in results.items():
        model = exp_data.get('model', 'unknown')
        loss = exp_data.get('loss_type', 'unknown')
        
        model_name = extract_model_name(model)
        loss_name = LOSS_DISPLAY_NAMES.get(loss, loss)
        
        if model_name not in model_results:
            model_results[model_name] = {}
        
        model_results[model_name][loss_name] = {
            'macro_f1': exp_data.get('macro_f1', 0) * 100,
            'no_hall_f1': exp_data.get('no_hallucination_f1', 0) * 100
        }
    
    if not model_results:
        print("No Phase 2 results found")
        return
    
    # 데이터프레임 생성
    losses = list(LOSS_DISPLAY_NAMES.values())
    
    for idx, (metric, title) in enumerate([('macro_f1', 'Macro F1'), ('no_hall_f1', 'no_hallucination F1')]):
        ax = axes[idx]
        
        x = np.arange(len(losses))
        width = 0.35
        multiplier = 0
        
        for model_name, loss_data in model_results.items():
            values = [loss_data.get(loss, {}).get(metric, 0) for loss in losses]
            offset = width * multiplier
            bars = ax.bar(x + offset, values, width, label=model_name)
            multiplier += 1
        
        ax.set_ylabel(f'{title} (%)', fontsize=12)
        ax.set_title(f'Loss Function Comparison - {title}', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (len(model_results) - 1) / 2)
        ax.set_xticklabels(losses, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim([90, 100])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_criterion_heatmap(results: Dict[str, Any], output_path: Path):
    """
    기준별 F1 히트맵
    """
    # 데이터 수집
    exp_names = []
    criterion_scores = {c: [] for c in CRITERIA}
    
    for exp_name, exp_data in results.items():
        exp_names.append(exp_name)
        for criterion in CRITERIA:
            score = exp_data.get(f'{criterion}_f1', 0) * 100
            criterion_scores[criterion].append(score)
    
    if not exp_names:
        print("No results found for heatmap")
        return
    
    # 데이터프레임 생성
    df = pd.DataFrame(criterion_scores, index=exp_names)
    
    # 히트맵 그리기
    fig, ax = plt.subplots(figsize=(14, max(8, len(exp_names) * 0.5)))
    
    sns.heatmap(
        df, 
        annot=True, 
        fmt='.1f', 
        cmap='RdYlGn',
        vmin=85, 
        vmax=100,
        ax=ax,
        cbar_kws={'label': 'F1 Score (%)'}
    )
    
    ax.set_title('Criterion-wise F1 Scores', fontsize=14, fontweight='bold')
    ax.set_xlabel('Evaluation Criteria', fontsize=12)
    ax.set_ylabel('Experiment', fontsize=12)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_final_ranking(all_results: Dict[str, Dict], output_path: Path):
    """
    최종 랭킹 테이블
    """
    # 모든 실험 결과 수집
    rankings = []
    
    for phase, results in all_results.items():
        for exp_name, exp_data in results.items():
            rankings.append({
                'Phase': phase,
                'Experiment': exp_name,
                'Model': extract_model_name(exp_data.get('model', 'unknown')),
                'Loss': LOSS_DISPLAY_NAMES.get(exp_data.get('loss_type', ''), exp_data.get('loss_type', '')),
                'Context': 'Yes' if exp_data.get('use_context', False) else 'No',
                'Macro F1': exp_data.get('macro_f1', 0) * 100,
                'no_hall F1': exp_data.get('no_hallucination_f1', 0) * 100
            })
    
    if not rankings:
        print("No results found for ranking")
        return
    
    df = pd.DataFrame(rankings)
    df = df.sort_values('Macro F1', ascending=False).head(15)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        colColours=['#3498db'] * len(df.columns)
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # 헤더 스타일
    for i in range(len(df.columns)):
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Top-3 행 하이라이트
    for i in range(1, min(4, len(df) + 1)):
        for j in range(len(df.columns)):
            table[(i, j)].set_facecolor('#d5f5e3')
    
    ax.set_title('Top 15 Experiment Results', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_phase_visualizations(phase: str, results: Dict[str, Any]):
    """
    특정 Phase의 시각화 생성
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    if phase == 'phase1_architecture' or phase == 'phase1':
        plot_architecture_comparison(results, FIGURES_DIR / '01_architecture_comparison.png')
        plot_context_effect(results, FIGURES_DIR / '02_context_effect.png')
    
    elif phase == 'phase2_loss' or phase == 'phase2':
        plot_loss_comparison(results, FIGURES_DIR / '03_loss_comparison.png')
    
    # 기준별 히트맵은 모든 Phase에서 생성
    plot_criterion_heatmap(results, FIGURES_DIR / f'04_criterion_heatmap_{phase}.png')


def generate_all_visualizations():
    """
    모든 Phase의 시각화 생성
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    # Phase별 결과 로드
    phase_files = [
        ('phase1', 'phase1_architecture_comparison.json'),
        ('phase2', 'phase2_loss_comparison.json'),
        ('phase3', 'phase3_advanced_architecture.json'),
        ('phase4', 'phase4_learning_strategy.json')
    ]
    
    for phase_name, filename in phase_files:
        results_path = RESULTS_DIR / filename
        if results_path.exists():
            results = load_results(results_path)
            all_results[phase_name] = results
            
            # Phase별 시각화
            generate_phase_visualizations(phase_name, results)
    
    # 최종 랭킹
    if all_results:
        plot_final_ranking(all_results, FIGURES_DIR / '05_final_ranking.png')
        
        # 최종 요약 저장
        summary = {
            'generated_at': datetime.now().isoformat(),
            'total_experiments': sum(len(r) for r in all_results.values()),
            'phases': list(all_results.keys())
        }
        save_results(summary, RESULTS_DIR / 'final_summary.json')


def main():
    parser = argparse.ArgumentParser(description='실험 결과 시각화')
    parser.add_argument('--phase', type=str, default=None,
                       help='시각화할 Phase (phase1, phase2, phase3, phase4)')
    parser.add_argument('--all', action='store_true',
                       help='모든 Phase 시각화')
    parser.add_argument('--results_dir', type=str, default=None,
                       help='결과 디렉토리 경로')
    parser.add_argument('--results_file', type=str, default=None,
                       help='단일 결과 파일 경로')
    
    args = parser.parse_args()
    
    if args.results_dir:
        global RESULTS_DIR
        RESULTS_DIR = Path(args.results_dir)
    
    if args.results_file:
        results = load_results(Path(args.results_file))
        phase = args.phase or 'custom'
        generate_phase_visualizations(phase, results)
    elif args.all:
        generate_all_visualizations()
    elif args.phase:
        # 특정 Phase 결과 로드
        phase_files = {
            'phase1': 'phase1_architecture_comparison.json',
            'phase2': 'phase2_loss_comparison.json',
            'phase3': 'phase3_advanced_architecture.json',
            'phase4': 'phase4_learning_strategy.json'
        }
        
        if args.phase in phase_files:
            results_path = RESULTS_DIR / phase_files[args.phase]
            if results_path.exists():
                results = load_results(results_path)
                generate_phase_visualizations(args.phase, results)
            else:
                print(f"Results file not found: {results_path}")
        else:
            print(f"Unknown phase: {args.phase}")
    else:
        print("Usage:")
        print("  python scripts/visualize_results.py --phase phase1")
        print("  python scripts/visualize_results.py --all")
        print("  python scripts/visualize_results.py --results_file path/to/results.json")


if __name__ == '__main__':
    main()
