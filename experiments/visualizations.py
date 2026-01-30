"""
Visualization Module for Video Anomaly Detection.

Generates comprehensive plots and visualizations for:
- ROC curves and PR curves
- Anomaly score timelines
- Model comparison charts
- Ablation study results
- Explainability visualizations
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import roc_curve, precision_recall_curve, auc

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


# ============================================================================
# COLOR SCHEMES
# ============================================================================

COLORS = {
    'baseline_mil': '#E74C3C',      # Red
    'mil_dsm': '#3498DB',            # Blue
    'mil_dsm_ra2r': '#2ECC71',       # Green
    'proposed_full': '#9B59B6',      # Purple
    'ablation_no_dsm': '#F39C12',    # Orange
    'ablation_no_ra2r': '#1ABC9C',   # Teal
    'ablation_no_gnn': '#E91E63',    # Pink
}

MODEL_LABELS = {
    'baseline_mil': 'Baseline (MIL)',
    'mil_dsm': 'MIL + DSM',
    'mil_dsm_ra2r': 'MIL + DSM + RA²R',
    'proposed_full': 'Proposed (Full)',
    'ablation_no_dsm': 'w/o DSM',
    'ablation_no_ra2r': 'w/o RA²R',
    'ablation_no_gnn': 'w/o GNN',
}


# ============================================================================
# ROC AND PR CURVES
# ============================================================================

def plot_roc_curves(
    results: Dict[str, Dict],
    predictions_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_path: str,
    figsize: Tuple[int, int] = (10, 8)
):
    """
    Plot ROC curves for multiple models.
    
    Args:
        results: Dictionary of experiment results
        predictions_data: Dict of {model_name: (predictions, labels)}
        save_path: Path to save the figure
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for model_name, (preds, labels) in predictions_data.items():
        fpr, tpr, _ = roc_curve(labels, preds)
        roc_auc = auc(fpr, tpr)
        
        color = COLORS.get(model_name, '#333333')
        label = f"{MODEL_LABELS.get(model_name, model_name)} (AUC = {roc_auc:.3f})"
        
        ax.plot(fpr, tpr, color=color, lw=2.5, label=label)
    
    # Diagonal line
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7, label='Random (AUC = 0.500)')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.set_title('ROC Curves - Model Comparison', fontsize=16, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC curves to: {save_path}")


def plot_pr_curves(
    predictions_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_path: str,
    figsize: Tuple[int, int] = (10, 8)
):
    """
    Plot Precision-Recall curves for multiple models.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for model_name, (preds, labels) in predictions_data.items():
        precision, recall, _ = precision_recall_curve(labels, preds)
        pr_auc = auc(recall, precision)
        
        color = COLORS.get(model_name, '#333333')
        label = f"{MODEL_LABELS.get(model_name, model_name)} (AUC = {pr_auc:.3f})"
        
        ax.plot(recall, precision, color=color, lw=2.5, label=label)
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=14)
    ax.set_ylabel('Precision', fontsize=14)
    ax.set_title('Precision-Recall Curves - Model Comparison', fontsize=16, fontweight='bold')
    ax.legend(loc='lower left', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved PR curves to: {save_path}")


# ============================================================================
# BAR CHARTS AND COMPARISON PLOTS
# ============================================================================

def plot_metrics_comparison(
    results: Dict[str, Dict],
    save_path: str,
    metrics: List[str] = ['roc_auc', 'pr_auc', 'precision', 'recall', 'f1_score'],
    figsize: Tuple[int, int] = (14, 8)
):
    """
    Create bar chart comparing metrics across models.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    models = list(results.keys())
    x = np.arange(len(metrics))
    width = 0.8 / len(models)
    
    for i, model in enumerate(models):
        values = [results[model]['metrics'].get(m, 0) for m in metrics]
        offset = (i - len(models) / 2 + 0.5) * width
        color = COLORS.get(model, f'C{i}')
        bars = ax.bar(x + offset, values, width, label=MODEL_LABELS.get(model, model),
                     color=color, alpha=0.85, edgecolor='black', linewidth=0.5)
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.annotate(f'{val:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8, rotation=90)
    
    ax.set_xlabel('Metrics', fontsize=14)
    ax.set_ylabel('Score', fontsize=14)
    ax.set_title('Performance Metrics Comparison Across Models', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved metrics comparison to: {save_path}")


def plot_ablation_results(
    results: Dict[str, Dict],
    save_path: str,
    figsize: Tuple[int, int] = (12, 6)
):
    """
    Create ablation study visualization.
    """
    # Filter ablation experiments
    ablation_models = {k: v for k, v in results.items() if 'ablation' in k or k == 'proposed_full'}
    
    if not ablation_models:
        print("No ablation results found.")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Component ablation
    component_ablations = ['proposed_full', 'ablation_no_dsm', 'ablation_no_ra2r', 'ablation_no_gnn']
    component_ablations = [m for m in component_ablations if m in results]
    
    if component_ablations:
        ax = axes[0]
        x = np.arange(len(component_ablations))
        aucs = [results[m]['metrics']['roc_auc'] for m in component_ablations]
        colors = [COLORS.get(m, 'gray') for m in component_ablations]
        labels = [MODEL_LABELS.get(m, m) for m in component_ablations]
        
        bars = ax.bar(x, aucs, color=colors, alpha=0.85, edgecolor='black')
        ax.set_ylabel('ROC-AUC', fontsize=12)
        ax.set_title('Component Ablation Study', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=10)
        ax.set_ylim(0.5, 1.0)
        ax.axhline(y=aucs[0], color='red', linestyle='--', alpha=0.5, label='Proposed')
        
        for bar, val in zip(bars, aucs):
            ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)
    
    # Segment ablation
    segment_ablations = [m for m in results.keys() if 'seg_' in m]
    if segment_ablations:
        ax = axes[1]
        segments = [int(m.split('_')[-1]) for m in segment_ablations]
        aucs = [results[m]['metrics']['roc_auc'] for m in segment_ablations]
        
        # Sort by segment count
        sorted_pairs = sorted(zip(segments, aucs))
        segments, aucs = zip(*sorted_pairs)
        
        ax.plot(segments, aucs, 'o-', color='#3498DB', markersize=10, linewidth=2)
        ax.set_xlabel('Number of Segments', fontsize=12)
        ax.set_ylabel('ROC-AUC', fontsize=12)
        ax.set_title('Segment Count Ablation', fontsize=14, fontweight='bold')
        ax.set_xticks(segments)
        ax.grid(True, alpha=0.3)
        
        for s, a in zip(segments, aucs):
            ax.annotate(f'{a:.3f}', xy=(s, a), xytext=(0, 10),
                       textcoords="offset points", ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved ablation results to: {save_path}")


# ============================================================================
# ANOMALY SCORE TIMELINE VISUALIZATION
# ============================================================================

def plot_anomaly_timeline(
    video_scores: np.ndarray,
    video_name: str,
    label: int,
    save_path: str,
    ground_truth_segments: Optional[List[Tuple[int, int]]] = None,
    figsize: Tuple[int, int] = (14, 5)
):
    """
    Plot anomaly score timeline for a single video.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    segments = np.arange(len(video_scores))
    
    # Color based on score intensity
    colors = plt.cm.RdYlGn_r(video_scores)
    
    # Plot bars
    bars = ax.bar(segments, video_scores, color=colors, alpha=0.8, edgecolor='none')
    
    # Plot line
    ax.plot(segments, video_scores, 'b-', linewidth=2, alpha=0.7, label='Anomaly Score')
    
    # Add threshold line
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Threshold (0.5)')
    
    # Highlight ground truth anomaly regions if provided
    if ground_truth_segments:
        for start, end in ground_truth_segments:
            ax.axvspan(start, end, alpha=0.2, color='red', label='Ground Truth Anomaly')
    
    label_text = "Abnormal Video" if label == 1 else "Normal Video"
    ax.set_xlabel('Segment Index', fontsize=12)
    ax.set_ylabel('Anomaly Score', fontsize=12)
    ax.set_title(f'Anomaly Score Timeline: {video_name}\n({label_text})', 
                fontsize=14, fontweight='bold')
    ax.set_xlim(-0.5, len(video_scores) - 0.5)
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper right')
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Anomaly Intensity', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_multiple_timelines(
    video_predictions: Dict[str, np.ndarray],
    labels_dict: Dict[str, int],
    save_dir: str,
    num_samples: int = 4
):
    """
    Plot anomaly timelines for multiple sample videos.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Select samples
    normal_videos = [v for v, l in labels_dict.items() if l == 0][:num_samples // 2]
    abnormal_videos = [v for v, l in labels_dict.items() if l == 1][:num_samples // 2]
    
    videos_to_plot = abnormal_videos + normal_videos
    
    # Create combined figure
    fig, axes = plt.subplots(len(videos_to_plot), 1, figsize=(14, 3 * len(videos_to_plot)))
    if len(videos_to_plot) == 1:
        axes = [axes]
    
    for ax, video_name in zip(axes, videos_to_plot):
        if video_name not in video_predictions:
            continue
        
        scores = video_predictions[video_name]
        label = labels_dict.get(video_name, -1)
        segments = np.arange(len(scores))
        
        colors = plt.cm.RdYlGn_r(scores)
        ax.bar(segments, scores, color=colors, alpha=0.8)
        ax.plot(segments, scores, 'b-', linewidth=1.5, alpha=0.7)
        ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
        
        label_text = "Abnormal" if label == 1 else "Normal"
        ax.set_title(f'{video_name} ({label_text})', fontsize=11, fontweight='bold')
        ax.set_xlim(-0.5, len(scores) - 0.5)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('Score')
        ax.grid(True, axis='y', alpha=0.3)
    
    axes[-1].set_xlabel('Segment Index')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'anomaly_timelines_combined.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved combined timelines to: {save_path}")


# ============================================================================
# TRAINING CURVES
# ============================================================================

def plot_training_curves(
    results: Dict[str, Dict],
    save_path: str,
    figsize: Tuple[int, int] = (12, 6)
):
    """
    Plot training loss curves for all models.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for model_name, result in results.items():
        if 'training_history' not in result:
            continue
        
        history = result['training_history']
        epochs = [h['epoch'] for h in history]
        losses = [h['loss'] for h in history]
        
        color = COLORS.get(model_name, f'C{list(results.keys()).index(model_name)}')
        label = MODEL_LABELS.get(model_name, model_name)
        
        ax.plot(epochs, losses, color=color, linewidth=2, label=label)
    
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Training Loss', fontsize=14)
    ax.set_title('Training Curves - All Models', fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved training curves to: {save_path}")


# ============================================================================
# RESULTS TABLE AS IMAGE
# ============================================================================

def create_results_table_image(
    results: Dict[str, Dict],
    save_path: str,
    figsize: Tuple[int, int] = (14, 8)
):
    """
    Create a publication-ready results table as an image.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    
    # Prepare data
    columns = ['Model', 'ROC-AUC', 'PR-AUC', 'Precision', 'Recall', 'F1-Score', 'Params']
    data = []
    
    for model_name, result in results.items():
        metrics = result['metrics']
        params = result.get('parameters', {}).get('trainable', 'N/A')
        if isinstance(params, int):
            params = f"{params:,}"
        
        row = [
            MODEL_LABELS.get(model_name, model_name),
            f"{metrics['roc_auc']:.4f}",
            f"{metrics['pr_auc']:.4f}",
            f"{metrics['precision']:.4f}",
            f"{metrics['recall']:.4f}",
            f"{metrics['f1_score']:.4f}",
            params
        ]
        data.append(row)
    
    # Create table
    table = ax.table(
        cellText=data,
        colLabels=columns,
        cellLoc='center',
        loc='center',
        colColours=['#4A90D9'] * len(columns)
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    
    # Style header
    for i in range(len(columns)):
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight best values
    for col_idx in range(1, 6):  # Metrics columns
        values = [float(data[row_idx][col_idx]) for row_idx in range(len(data))]
        best_idx = np.argmax(values)
        table[(best_idx + 1, col_idx)].set_facecolor('#90EE90')
    
    plt.title('Video Anomaly Detection - Results Comparison', 
             fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved results table to: {save_path}")


# ============================================================================
# MAIN VISUALIZATION FUNCTION
# ============================================================================

def generate_all_visualizations(results_path: str, output_dir: str = None):
    """
    Generate all visualizations from experiment results.
    
    Args:
        results_path: Path to the combined results JSON file
        output_dir: Output directory for visualizations
    """
    # Load results
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    if output_dir is None:
        output_dir = os.path.dirname(results_path)
    
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("Generating Visualizations")
    print("=" * 60)
    
    # 1. Metrics comparison bar chart
    plot_metrics_comparison(
        results,
        os.path.join(viz_dir, 'metrics_comparison.png')
    )
    
    # 2. Ablation study results
    plot_ablation_results(
        results,
        os.path.join(viz_dir, 'ablation_results.png')
    )
    
    # 3. Training curves
    plot_training_curves(
        results,
        os.path.join(viz_dir, 'training_curves.png')
    )
    
    # 4. Results table image
    create_results_table_image(
        results,
        os.path.join(viz_dir, 'results_table.png')
    )
    
    print(f"\nAll visualizations saved to: {viz_dir}")


# ============================================================================
# SIMULATE RESULTS FOR DEMONSTRATION
# ============================================================================

def generate_demo_results():
    """
    Generate simulated results for demonstration purposes.
    """
    np.random.seed(42)
    
    results = {
        'baseline_mil': {
            'experiment': {'name': 'baseline_mil'},
            'metrics': {
                'roc_auc': 0.7523,
                'pr_auc': 0.6845,
                'precision': 0.6234,
                'recall': 0.7012,
                'f1_score': 0.6600,
                'accuracy': 0.6890
            },
            'parameters': {'trainable': 1245678},
            'training_history': [{'epoch': i, 'loss': 1.5 - 0.02 * i + np.random.normal(0, 0.05)} for i in range(1, 51)]
        },
        'mil_dsm': {
            'experiment': {'name': 'mil_dsm'},
            'metrics': {
                'roc_auc': 0.7891,
                'pr_auc': 0.7234,
                'precision': 0.6589,
                'recall': 0.7345,
                'f1_score': 0.6947,
                'accuracy': 0.7210
            },
            'parameters': {'trainable': 1567890},
            'training_history': [{'epoch': i, 'loss': 1.4 - 0.022 * i + np.random.normal(0, 0.04)} for i in range(1, 51)]
        },
        'mil_dsm_ra2r': {
            'experiment': {'name': 'mil_dsm_ra2r'},
            'metrics': {
                'roc_auc': 0.8234,
                'pr_auc': 0.7612,
                'precision': 0.6923,
                'recall': 0.7689,
                'f1_score': 0.7286,
                'accuracy': 0.7534
            },
            'parameters': {'trainable': 1890123},
            'training_history': [{'epoch': i, 'loss': 1.35 - 0.023 * i + np.random.normal(0, 0.035)} for i in range(1, 51)]
        },
        'proposed_full': {
            'experiment': {'name': 'proposed_full'},
            'metrics': {
                'roc_auc': 0.8567,
                'pr_auc': 0.7923,
                'precision': 0.7234,
                'recall': 0.7912,
                'f1_score': 0.7558,
                'accuracy': 0.7823
            },
            'parameters': {'trainable': 2156789},
            'training_history': [{'epoch': i, 'loss': 1.3 - 0.025 * i + np.random.normal(0, 0.03)} for i in range(1, 51)]
        },
        'ablation_no_dsm': {
            'experiment': {'name': 'ablation_no_dsm'},
            'metrics': {
                'roc_auc': 0.8123,
                'pr_auc': 0.7456,
                'precision': 0.6789,
                'recall': 0.7623,
                'f1_score': 0.7182,
                'accuracy': 0.7412
            },
            'parameters': {'trainable': 1789012},
            'training_history': [{'epoch': i, 'loss': 1.38 - 0.024 * i + np.random.normal(0, 0.04)} for i in range(1, 51)]
        },
        'ablation_no_ra2r': {
            'experiment': {'name': 'ablation_no_ra2r'},
            'metrics': {
                'roc_auc': 0.8234,
                'pr_auc': 0.7534,
                'precision': 0.6912,
                'recall': 0.7534,
                'f1_score': 0.7210,
                'accuracy': 0.7489
            },
            'parameters': {'trainable': 1923456},
            'training_history': [{'epoch': i, 'loss': 1.36 - 0.024 * i + np.random.normal(0, 0.035)} for i in range(1, 51)]
        },
        'ablation_no_gnn': {
            'experiment': {'name': 'ablation_no_gnn'},
            'metrics': {
                'roc_auc': 0.8312,
                'pr_auc': 0.7623,
                'precision': 0.7012,
                'recall': 0.7689,
                'f1_score': 0.7335,
                'accuracy': 0.7567
            },
            'parameters': {'trainable': 1678901},
            'training_history': [{'epoch': i, 'loss': 1.34 - 0.024 * i + np.random.normal(0, 0.035)} for i in range(1, 51)]
        },
    }
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Visualizations')
    parser.add_argument('--results', type=str, default=None,
                       help='Path to results JSON file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory')
    parser.add_argument('--demo', action='store_true',
                       help='Generate demo visualizations with simulated data')
    
    args = parser.parse_args()
    
    if args.demo:
        # Generate demo results and visualizations
        results = generate_demo_results()
        output_dir = os.path.join(project_root, 'experiments', 'results', 'demo_visualizations')
        os.makedirs(output_dir, exist_ok=True)
        
        # Save demo results
        results_path = os.path.join(output_dir, 'demo_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        generate_all_visualizations(results_path, output_dir)
    elif args.results:
        generate_all_visualizations(args.results, args.output)
    else:
        print("Please provide --results path or use --demo for demonstration.")
