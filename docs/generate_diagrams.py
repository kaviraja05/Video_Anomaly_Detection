#!/usr/bin/env python3
"""
Generate Publication-Quality Architecture Diagrams
This script creates professional diagrams for the research paper and presentations.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.collections import PatchCollection
import numpy as np

# Set publication-quality defaults
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})


def create_architecture_diagram():
    """Create the main model architecture diagram."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Colors
    colors = {
        'input': '#E3F2FD',     # Light blue
        'embed': '#BBDEFB',     # Blue
        'temporal': '#90CAF9',  # Darker blue
        'dsm': '#81D4FA',       # Cyan
        'gnn': '#80DEEA',       # Teal
        'ra2r': '#A5D6A7',      # Green
        'scorer': '#C8E6C9',    # Light green
        'output': '#FFF9C4',    # Yellow
        'arrow': '#424242'      # Dark gray
    }
    
    # Box dimensions
    box_width = 3.5
    box_height = 0.8
    y_positions = [8.5, 7.2, 5.9, 4.6, 3.3, 2.0, 0.7]
    x_center = 7
    
    def draw_box(y, label, sublabel, color, x=None):
        if x is None:
            x = x_center - box_width/2
        box = FancyBboxPatch((x, y), box_width, box_height,
                             boxstyle="round,pad=0.05,rounding_size=0.1",
                             facecolor=color, edgecolor='#333333', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + box_width/2, y + box_height/2 + 0.1, label,
                ha='center', va='center', fontsize=11, fontweight='bold')
        ax.text(x + box_width/2, y + box_height/2 - 0.2, sublabel,
                ha='center', va='center', fontsize=9, style='italic', color='#555555')
    
    def draw_arrow(y1, y2, x=x_center):
        ax.annotate('', xy=(x, y2 + box_height), xytext=(x, y1),
                    arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2))
    
    # Draw boxes
    draw_box(y_positions[0], 'Input Video', 'T frames × H × W × 3', colors['input'])
    draw_box(y_positions[1], 'Feature Embedding', '2048-D → 512-D → 128-D', colors['embed'])
    draw_box(y_positions[2], 'Temporal Module', 'Multi-scale Conv1D (k=3,5)', colors['temporal'])
    draw_box(y_positions[3], 'Dynamic Similarity (DSM)', 'Attention-based Adjacency', colors['dsm'])
    draw_box(y_positions[4], 'Graph Neural Network', '2 layers, 4 attention heads', colors['gnn'])
    draw_box(y_positions[5], 'RA²R Module', 'Relation-Aware Reasoning', colors['ra2r'])
    draw_box(y_positions[6], 'Anomaly Scorer', 'MLP + Sigmoid → [0,1]', colors['scorer'])
    
    # Draw arrows
    for i in range(len(y_positions) - 1):
        draw_arrow(y_positions[i], y_positions[i+1])
    
    # Add side annotations
    annotations = [
        (y_positions[1], 'LayerNorm + Dropout'),
        (y_positions[2], 'Residual Connection'),
        (y_positions[3], 'Dynamic Graph Construction'),
        (y_positions[4], 'Message Passing'),
        (y_positions[5], 'Pairwise Relation Encoding'),
    ]
    
    for y, text in annotations:
        ax.text(x_center + box_width/2 + 0.5, y + box_height/2, text,
                ha='left', va='center', fontsize=9, color='#666666')
    
    # Title
    ax.text(x_center, 9.5, 'Proposed Model Architecture',
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Add dimension annotations on left
    dims = [
        (y_positions[0], '(N×2048)'),
        (y_positions[1], '(N×128)'),
        (y_positions[2], '(N×128)'),
        (y_positions[3], '(N×N)'),
        (y_positions[4], '(N×128)'),
        (y_positions[5], '(N×128)'),
        (y_positions[6], '(N×1)'),
    ]
    
    for y, dim in dims:
        ax.text(x_center - box_width/2 - 0.3, y + box_height/2, dim,
                ha='right', va='center', fontsize=9, color='#666666', family='monospace')
    
    plt.savefig('architecture_main.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('architecture_main.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Created architecture_main.png/pdf")


def create_dsm_diagram():
    """Create detailed DSM module diagram."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    colors = {
        'input': '#E8F5E9',
        'query': '#BBDEFB',
        'key': '#B3E5FC',
        'attention': '#FFE0B2',
        'gate': '#F8BBD9',
        'output': '#C8E6C9'
    }
    
    # Input
    ax.add_patch(FancyBboxPatch((5, 7), 2, 0.6, boxstyle="round,pad=0.05",
                                facecolor=colors['input'], edgecolor='black'))
    ax.text(6, 7.3, 'Input Features h', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(6, 7.05, '(N × d)', ha='center', va='center', fontsize=8)
    
    # Query and Key branches
    ax.annotate('', xy=(4, 5.8), xytext=(5.5, 6.8),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.annotate('', xy=(8, 5.8), xytext=(6.5, 6.8),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    
    ax.add_patch(FancyBboxPatch((3, 5.2), 2, 0.6, boxstyle="round,pad=0.05",
                                facecolor=colors['query'], edgecolor='black'))
    ax.text(4, 5.5, 'Query (Wq)', ha='center', va='center', fontsize=10)
    
    ax.add_patch(FancyBboxPatch((7, 5.2), 2, 0.6, boxstyle="round,pad=0.05",
                                facecolor=colors['key'], edgecolor='black'))
    ax.text(8, 5.5, 'Key (Wk)', ha='center', va='center', fontsize=10)
    
    # Attention computation
    ax.annotate('', xy=(5.5, 4.3), xytext=(4, 5.0),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.annotate('', xy=(6.5, 4.3), xytext=(8, 5.0),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    
    ax.add_patch(FancyBboxPatch((4.5, 3.7), 3, 0.6, boxstyle="round,pad=0.05",
                                facecolor=colors['attention'], edgecolor='black'))
    ax.text(6, 4.0, 'Attention = Q·Kᵀ/√d', ha='center', va='center', fontsize=10)
    
    # Context gate
    ax.annotate('', xy=(6, 3.1), xytext=(6, 3.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    
    ax.add_patch(FancyBboxPatch((4.5, 2.5), 3, 0.6, boxstyle="round,pad=0.05",
                                facecolor=colors['gate'], edgecolor='black'))
    ax.text(6, 2.8, 'Context Gate σ(W·[hi;hj])', ha='center', va='center', fontsize=10)
    
    # Threshold
    ax.annotate('', xy=(6, 1.9), xytext=(6, 2.3),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    
    ax.add_patch(FancyBboxPatch((4.5, 1.3), 3, 0.6, boxstyle="round,pad=0.05",
                                facecolor='#FFCCBC', edgecolor='black'))
    ax.text(6, 1.6, 'Threshold (τ)', ha='center', va='center', fontsize=10)
    
    # Output
    ax.annotate('', xy=(6, 0.7), xytext=(6, 1.1),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    
    ax.add_patch(FancyBboxPatch((4.5, 0.2), 3, 0.5, boxstyle="round,pad=0.05",
                                facecolor=colors['output'], edgecolor='black'))
    ax.text(6, 0.45, 'Adjacency Matrix A', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Title
    ax.text(6, 7.8, 'Dynamic Similarity Module (DSM)', ha='center', va='center',
            fontsize=14, fontweight='bold')
    
    # Add adjacency visualization
    ax.add_patch(FancyBboxPatch((9.5, 1.5), 2, 2, boxstyle="round,pad=0.05",
                                facecolor='white', edgecolor='black'))
    ax.text(10.5, 3.7, 'Adjacency A', ha='center', va='center', fontsize=9)
    
    # Draw small grid for adjacency
    for i in range(4):
        for j in range(4):
            val = np.random.rand()
            color = plt.cm.Blues(val)
            ax.add_patch(plt.Rectangle((9.6 + j*0.45, 1.6 + (3-i)*0.45), 0.4, 0.4,
                                       facecolor=color, edgecolor='gray', linewidth=0.5))
    
    plt.savefig('dsm_module.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Created dsm_module.png")


def create_gnn_diagram():
    """Create GNN layer diagram with attention mechanism."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Draw nodes in a graph
    node_positions = [
        (3, 6), (5, 6.5), (4, 4.5), (6, 4), (2, 4)
    ]
    
    # Draw edges with attention weights
    edges = [
        (0, 1, 0.8), (0, 2, 0.6), (0, 4, 0.9),
        (1, 2, 0.7), (1, 3, 0.5),
        (2, 3, 0.8), (2, 4, 0.4),
        (3, 4, 0.3)
    ]
    
    # Draw edges first
    for i, j, w in edges:
        x1, y1 = node_positions[i]
        x2, y2 = node_positions[j]
        ax.plot([x1, x2], [y1, y2], color=plt.cm.Blues(w), linewidth=2*w+1,
                alpha=0.7, zorder=1)
    
    # Draw nodes
    for i, (x, y) in enumerate(node_positions):
        circle = Circle((x, y), 0.3, facecolor='#81D4FA', edgecolor='#01579B',
                        linewidth=2, zorder=2)
        ax.add_patch(circle)
        ax.text(x, y, f'$h_{i+1}$', ha='center', va='center', fontsize=10, zorder=3)
    
    # Draw the message passing illustration
    ax.text(4, 7.3, 'Graph Neural Network Layer', ha='center', va='center',
            fontsize=14, fontweight='bold')
    
    # Add formulas box
    formula_box = FancyBboxPatch((7, 3), 4.5, 4, boxstyle="round,pad=0.1",
                                  facecolor='#FFF3E0', edgecolor='#FF9800', linewidth=2)
    ax.add_patch(formula_box)
    
    formulas = [
        (7.2, 6.5, 'Multi-Head Attention:', True),
        (7.2, 6.0, r'$\alpha_{ij} = \mathrm{softmax}(\frac{Q_i K_j^T}{\sqrt{d}})$', False),
        (7.2, 5.3, 'Message Aggregation:', True),
        (7.2, 4.8, r"$h'_i = \sum_j \alpha_{ij} \cdot V_j$", False),
        (7.2, 4.1, 'Update with Residual:', True),
        (7.2, 3.6, r"$h''_i = \mathrm{LN}(h_i + h'_i)$", False),
    ]
    
    for x, y, text, is_title in formulas:
        weight = 'bold' if is_title else 'normal'
        size = 10 if is_title else 9
        ax.text(x, y, text, ha='left', va='center', fontsize=size, fontweight=weight)
    
    # Add legend for edge colors
    ax.text(1.5, 2.5, 'Edge Color = Attention Weight', ha='left', fontsize=9)
    
    # Draw color bar
    for i, val in enumerate([0.3, 0.5, 0.7, 0.9]):
        ax.add_patch(plt.Rectangle((1.5 + i*0.6, 2.0), 0.5, 0.3,
                                   facecolor=plt.cm.Blues(val), edgecolor='gray'))
        ax.text(1.75 + i*0.6, 1.8, f'{val}', ha='center', fontsize=8)
    
    plt.savefig('gnn_layer.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Created gnn_layer.png")


def create_mil_loss_diagram():
    """Create MIL loss illustration."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Title
    ax.text(6, 5.7, 'Multiple Instance Learning Loss', ha='center', va='center',
            fontsize=14, fontweight='bold')
    
    # Draw normal bag
    ax.add_patch(FancyBboxPatch((0.5, 2.5), 4.5, 2.5, boxstyle="round,pad=0.1",
                                facecolor='#E8F5E9', edgecolor='#4CAF50', linewidth=2))
    ax.text(2.75, 4.7, 'Normal Bag (Label: 0)', ha='center', va='center',
            fontsize=11, fontweight='bold', color='#2E7D32')
    
    # Normal instances (all low scores)
    normal_scores = [0.1, 0.15, 0.08, 0.12, 0.18, 0.05, 0.11, 0.09]
    for i, score in enumerate(normal_scores):
        x = 0.8 + (i % 4) * 1.0
        y = 3.6 if i < 4 else 2.9
        color = plt.cm.Greens(0.3 + score)
        ax.add_patch(Circle((x, y), 0.2, facecolor=color, edgecolor='#388E3C'))
        ax.text(x, y, f'{score:.2f}', ha='center', va='center', fontsize=7)
    
    # Draw abnormal bag
    ax.add_patch(FancyBboxPatch((6.5, 2.5), 4.5, 2.5, boxstyle="round,pad=0.1",
                                facecolor='#FFEBEE', edgecolor='#F44336', linewidth=2))
    ax.text(8.75, 4.7, 'Abnormal Bag (Label: 1)', ha='center', va='center',
            fontsize=11, fontweight='bold', color='#C62828')
    
    # Abnormal instances (mixed, some high)
    abnormal_scores = [0.12, 0.85, 0.78, 0.15, 0.92, 0.11, 0.88, 0.14]
    for i, score in enumerate(abnormal_scores):
        x = 6.8 + (i % 4) * 1.0
        y = 3.6 if i < 4 else 2.9
        color = plt.cm.Reds(0.2 + score * 0.6)
        ax.add_patch(Circle((x, y), 0.2, facecolor=color, edgecolor='#D32F2F'))
        ax.text(x, y, f'{score:.2f}', ha='center', va='center', fontsize=7)
    
    # Draw loss formula
    ax.add_patch(FancyBboxPatch((1.5, 0.3), 9, 1.5, boxstyle="round,pad=0.1",
                                facecolor='#FFF8E1', edgecolor='#FFA000', linewidth=2))
    ax.text(6, 1.5, 'MIL Ranking Loss:', ha='center', va='center',
            fontsize=11, fontweight='bold')
    ax.text(6, 0.9, r'$\mathcal{L} = \max(0, 1 - \max(S_{abn}) + \max(S_{norm})) + \lambda_{smooth} + \lambda_{sparse}$',
            ha='center', va='center', fontsize=10)
    
    # Arrows pointing to max scores
    ax.annotate('max = 0.18', xy=(2.8, 2.9), xytext=(2.8, 2.2),
                arrowprops=dict(arrowstyle='->', color='#388E3C', lw=1.5),
                fontsize=9, color='#2E7D32', ha='center')
    
    ax.annotate('max = 0.92', xy=(8.8, 2.9), xytext=(8.8, 2.2),
                arrowprops=dict(arrowstyle='->', color='#D32F2F', lw=1.5),
                fontsize=9, color='#C62828', ha='center')
    
    plt.savefig('mil_loss.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Created mil_loss.png")


def create_workflow_diagram():
    """Create end-to-end workflow diagram."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis('off')
    
    # Stage positions
    stages = [
        (1, 'Video\nInput', '#E3F2FD'),
        (3, 'I3D\nExtraction', '#BBDEFB'),
        (5, 'Segment\n(N=32)', '#90CAF9'),
        (7, 'GNN\nModel', '#80DEEA'),
        (9, 'Anomaly\nScores', '#A5D6A7'),
        (11, 'Detection\nResult', '#C8E6C9'),
    ]
    
    # Draw stages
    for x, label, color in stages:
        box = FancyBboxPatch((x - 0.8, 1.2), 1.6, 1.6,
                             boxstyle="round,pad=0.05,rounding_size=0.15",
                             facecolor=color, edgecolor='#333333', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, 2, label, ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Draw arrows
    for i in range(len(stages) - 1):
        x1 = stages[i][0] + 0.9
        x2 = stages[i+1][0] - 0.9
        ax.annotate('', xy=(x2, 2), xytext=(x1, 2),
                    arrowprops=dict(arrowstyle='->', color='#424242', lw=2))
    
    # Title
    ax.text(6, 3.5, 'End-to-End Video Anomaly Detection Pipeline',
            ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Add timing annotations
    timings = ['', '~2s/video', '<0.1s', '~0.05s', '', '']
    for i, (x, _, _) in enumerate(stages):
        ax.text(x, 0.8, timings[i], ha='center', va='center',
                fontsize=8, color='#666666', style='italic')
    
    plt.savefig('workflow.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Created workflow.png")


def create_results_comparison():
    """Create results comparison chart."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Data
    methods = ['SVM', 'C3D+\nMIL', 'Sultani\net al.', 'RTFM', 'Ours']
    auc_scores = [50.0, 75.4, 77.9, 84.3, 86.7]
    colors = ['#BDBDBD', '#90CAF9', '#81D4FA', '#80DEEA', '#4CAF50']
    
    # AUC-ROC comparison
    ax1 = axes[0]
    bars1 = ax1.bar(methods, auc_scores, color=colors, edgecolor='black', linewidth=1)
    ax1.set_ylabel('AUC-ROC (%)', fontsize=12)
    ax1.set_title('(a) Comparison with State-of-the-Art', fontsize=12, fontweight='bold')
    ax1.set_ylim(40, 95)
    ax1.axhline(y=86.7, color='#2E7D32', linestyle='--', linewidth=1, alpha=0.7)
    
    # Add value labels
    for bar, val in zip(bars1, auc_scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10)
    
    # Ablation study
    ax2 = axes[1]
    ablation_methods = ['Base\n(MIL)', '+DSM', '+GNN', '+RA²R\n(Full)']
    ablation_scores = [77.4, 80.2, 83.5, 86.7]
    ablation_colors = ['#FFCDD2', '#FFE0B2', '#C8E6C9', '#4CAF50']
    
    bars2 = ax2.bar(ablation_methods, ablation_scores, color=ablation_colors,
                    edgecolor='black', linewidth=1)
    ax2.set_ylabel('AUC-ROC (%)', fontsize=12)
    ax2.set_title('(b) Ablation Study', fontsize=12, fontweight='bold')
    ax2.set_ylim(70, 92)
    
    # Add improvement annotations
    improvements = ['', '+2.8', '+3.3', '+3.2']
    for bar, val, imp in zip(bars2, ablation_scores, improvements):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10)
        if imp:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 2,
                    imp, ha='center', va='top', fontsize=9, color='#2E7D32',
                    fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results_comparison.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Created results_comparison.png")


def create_segment_visualization():
    """Create segment-level anomaly score visualization."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    
    # Generate example data
    np.random.seed(42)
    segments = np.arange(32)
    
    # Normal video
    normal_scores = np.random.rand(32) * 0.3
    normal_scores = np.convolve(normal_scores, np.ones(3)/3, mode='same')
    
    # Abnormal video (with anomaly in segments 10-18)
    abnormal_scores = np.random.rand(32) * 0.3
    abnormal_scores[10:18] = 0.7 + np.random.rand(8) * 0.25
    abnormal_scores = np.convolve(abnormal_scores, np.ones(3)/3, mode='same')
    
    # Plot normal video
    ax1 = axes[0]
    ax1.fill_between(segments, normal_scores, alpha=0.3, color='#4CAF50')
    ax1.plot(segments, normal_scores, 'o-', color='#2E7D32', linewidth=2, markersize=4)
    ax1.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, label='Threshold')
    ax1.set_ylabel('Anomaly Score', fontsize=11)
    ax1.set_title('Normal Video - All scores below threshold', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 1)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot abnormal video
    ax2 = axes[1]
    colors = ['#F44336' if s > 0.5 else '#4CAF50' for s in abnormal_scores]
    ax2.fill_between(segments, abnormal_scores, alpha=0.3, color='#FF9800')
    ax2.plot(segments, abnormal_scores, 'o-', color='#E65100', linewidth=2, markersize=4)
    ax2.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, label='Threshold')
    ax2.axvspan(10, 17, alpha=0.2, color='red', label='Detected Anomaly')
    ax2.set_xlabel('Segment Index', fontsize=11)
    ax2.set_ylabel('Anomaly Score', fontsize=11)
    ax2.set_title('Abnormal Video - Anomaly detected in segments 10-17', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('segment_visualization.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Created segment_visualization.png")


def main():
    """Generate all diagrams."""
    print("Generating publication-quality diagrams...\n")
    
    create_architecture_diagram()
    create_dsm_diagram()
    create_gnn_diagram()
    create_mil_loss_diagram()
    create_workflow_diagram()
    create_results_comparison()
    create_segment_visualization()
    
    print("\n" + "="*50)
    print("All diagrams generated successfully!")
    print("="*50)
    print("\nGenerated files:")
    print("  - architecture_main.png/pdf")
    print("  - dsm_module.png")
    print("  - gnn_layer.png")
    print("  - mil_loss.png")
    print("  - workflow.png")
    print("  - results_comparison.png")
    print("  - segment_visualization.png")


if __name__ == '__main__':
    main()
