"""
Explainability Module for Video Anomaly Detection.

Provides interpretable visualizations and explanations for:
- Segment importance / attention weights
- GNN graph relationships
- Feature contributions
- Decision reasoning
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

plt.style.use('seaborn-v0_8-whitegrid')


class ModelExplainer:
    """
    Explainability module for the Video Anomaly Detection model.
    
    Provides methods to:
    1. Extract attention weights from DSM and GNN
    2. Compute segment importance scores
    3. Visualize graph relationships
    4. Generate human-readable explanations
    """
    
    def __init__(self, model: nn.Module, device: torch.device = None):
        """
        Initialize the explainer.
        
        Args:
            model: Trained ProposedModel
            device: Computation device
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval()
        
        # Storage for intermediate activations
        self.activations = {}
        self.gradients = {}
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture intermediate activations."""
        def get_activation(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    self.activations[name] = output[0].detach()
                else:
                    self.activations[name] = output.detach()
            return hook
        
        # Register hooks on key modules
        if hasattr(self.model, 'dsm'):
            self.model.dsm.register_forward_hook(get_activation('dsm'))
        if hasattr(self.model, 'gnn'):
            self.model.gnn.register_forward_hook(get_activation('gnn'))
        if hasattr(self.model, 'ra2r'):
            self.model.ra2r.register_forward_hook(get_activation('ra2r'))
        if hasattr(self.model, 'scorer'):
            self.model.scorer.register_forward_hook(get_activation('scorer'))
    
    @torch.no_grad()
    def explain_video(
        self,
        features: torch.Tensor,
        video_name: str = "video"
    ) -> Dict:
        """
        Generate explanation for a single video.
        
        Args:
            features: Video features tensor of shape (1, T, D)
            video_name: Name of the video
            
        Returns:
            Dictionary containing explanation data
        """
        features = features.to(self.device)
        
        # Forward pass with feature extraction
        output = self.model(features, return_features=True)
        
        scores = output['scores'].cpu().numpy()[0]  # (T,)
        
        explanation = {
            'video_name': video_name,
            'anomaly_scores': scores.tolist(),
            'max_score': float(scores.max()),
            'mean_score': float(scores.mean()),
            'prediction': 'Abnormal' if scores.max() > 0.5 else 'Normal',
            'segment_analysis': self._analyze_segments(scores),
        }
        
        # Extract attention/adjacency if available
        if 'adjacency' in output:
            adj = output['adjacency'].cpu().numpy()[0]  # (T, T)
            explanation['adjacency_matrix'] = adj.tolist()
            explanation['graph_analysis'] = self._analyze_graph(adj, scores)
        
        if 'segment_features' in output:
            seg_features = output['segment_features'].cpu().numpy()[0]
            explanation['feature_norms'] = np.linalg.norm(seg_features, axis=1).tolist()
        
        return explanation
    
    def _analyze_segments(self, scores: np.ndarray) -> Dict:
        """Analyze segment importance."""
        n_segments = len(scores)
        
        # Find peaks (local maxima)
        peaks = []
        for i in range(1, n_segments - 1):
            if scores[i] > scores[i-1] and scores[i] > scores[i+1]:
                peaks.append(i)
        
        # Find anomalous regions (continuous segments above threshold)
        threshold = 0.5
        anomalous_regions = []
        in_region = False
        start = 0
        
        for i, score in enumerate(scores):
            if score > threshold and not in_region:
                start = i
                in_region = True
            elif score <= threshold and in_region:
                anomalous_regions.append({
                    'start': start,
                    'end': i - 1,
                    'max_score': float(scores[start:i].max()),
                    'mean_score': float(scores[start:i].mean())
                })
                in_region = False
        
        if in_region:
            anomalous_regions.append({
                'start': start,
                'end': n_segments - 1,
                'max_score': float(scores[start:].max()),
                'mean_score': float(scores[start:].mean())
            })
        
        # Top-k important segments
        topk_indices = np.argsort(scores)[-5:][::-1]
        
        return {
            'peaks': peaks,
            'anomalous_regions': anomalous_regions,
            'top_segments': topk_indices.tolist(),
            'top_scores': scores[topk_indices].tolist()
        }
    
    def _analyze_graph(self, adj: np.ndarray, scores: np.ndarray) -> Dict:
        """Analyze graph structure from adjacency matrix."""
        # Threshold adjacency for analysis
        adj_binary = (adj > 0.5).astype(int)
        
        # Compute degree of each node
        degrees = adj_binary.sum(axis=1)
        
        # Find highly connected segments
        high_degree_segments = np.where(degrees > np.mean(degrees) + np.std(degrees))[0]
        
        # Correlation between connectivity and anomaly score
        degree_score_corr = np.corrcoef(degrees, scores)[0, 1] if len(scores) > 1 else 0
        
        return {
            'mean_degree': float(degrees.mean()),
            'high_connectivity_segments': high_degree_segments.tolist(),
            'degree_score_correlation': float(degree_score_corr),
            'graph_density': float(adj_binary.sum() / (len(adj) ** 2))
        }
    
    def visualize_segment_importance(
        self,
        explanation: Dict,
        save_path: str,
        figsize: Tuple[int, int] = (14, 6)
    ):
        """
        Visualize segment importance with detailed annotations.
        """
        fig, axes = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
        
        scores = np.array(explanation['anomaly_scores'])
        n_segments = len(scores)
        x = np.arange(n_segments)
        
        # Main plot - anomaly scores
        ax1 = axes[0]
        colors = plt.cm.RdYlGn_r(scores)
        bars = ax1.bar(x, scores, color=colors, alpha=0.8, edgecolor='none', width=0.9)
        ax1.plot(x, scores, 'b-', linewidth=2, alpha=0.7, marker='o', markersize=4)
        ax1.axhline(y=0.5, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Threshold')
        
        # Highlight top segments
        top_segments = explanation['segment_analysis']['top_segments']
        for seg in top_segments[:3]:
            ax1.annotate(f'Top\n{scores[seg]:.2f}',
                        xy=(seg, scores[seg]), xytext=(seg, scores[seg] + 0.1),
                        ha='center', fontsize=9, fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))
        
        # Mark anomalous regions
        for region in explanation['segment_analysis']['anomalous_regions']:
            ax1.axvspan(region['start'] - 0.5, region['end'] + 0.5, 
                       alpha=0.15, color='red', label='Anomaly Region')
        
        ax1.set_xlim(-0.5, n_segments - 0.5)
        ax1.set_ylim(0, 1.2)
        ax1.set_ylabel('Anomaly Score', fontsize=12)
        ax1.set_title(f"Segment Importance Analysis: {explanation['video_name']}\n"
                     f"Prediction: {explanation['prediction']} (Max Score: {explanation['max_score']:.3f})",
                     fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, axis='y', alpha=0.3)
        
        # Bottom plot - segment classification
        ax2 = axes[1]
        segment_colors = ['#2ECC71' if s < 0.3 else '#F39C12' if s < 0.5 else '#E74C3C' for s in scores]
        ax2.bar(x, [1] * n_segments, color=segment_colors, alpha=0.8, width=0.9)
        ax2.set_xlim(-0.5, n_segments - 0.5)
        ax2.set_ylim(0, 1)
        ax2.set_xlabel('Segment Index', fontsize=12)
        ax2.set_ylabel('Status', fontsize=10)
        ax2.set_yticks([])
        
        # Legend for segment classification
        legend_elements = [
            mpatches.Patch(color='#2ECC71', label='Normal (<0.3)'),
            mpatches.Patch(color='#F39C12', label='Suspicious (0.3-0.5)'),
            mpatches.Patch(color='#E74C3C', label='Anomalous (>0.5)')
        ]
        ax2.legend(handles=legend_elements, loc='upper right', ncol=3, fontsize=9)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved segment importance visualization to: {save_path}")
    
    def visualize_graph_relationships(
        self,
        explanation: Dict,
        save_path: str,
        figsize: Tuple[int, int] = (12, 10)
    ):
        """
        Visualize GNN graph relationships between segments.
        """
        if 'adjacency_matrix' not in explanation:
            print("No adjacency matrix available for graph visualization.")
            return
        
        adj = np.array(explanation['adjacency_matrix'])
        scores = np.array(explanation['anomaly_scores'])
        n_segments = len(scores)
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], width_ratios=[1, 1])
        
        # Graph visualization
        ax1 = fig.add_subplot(gs[0, :])
        
        # Create networkx graph
        G = nx.Graph()
        for i in range(n_segments):
            G.add_node(i, score=scores[i])
        
        # Add edges based on adjacency
        threshold = 0.5
        for i in range(n_segments):
            for j in range(i + 1, n_segments):
                if adj[i, j] > threshold:
                    G.add_edge(i, j, weight=adj[i, j])
        
        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # Node colors based on anomaly scores
        node_colors = [plt.cm.RdYlGn_r(scores[i]) for i in range(n_segments)]
        node_sizes = [300 + 500 * scores[i] for i in range(n_segments)]
        
        # Draw graph
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                               alpha=0.8, ax=ax1)
        nx.draw_networkx_edges(G, pos, alpha=0.3, width=1, ax=ax1)
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax1)
        
        ax1.set_title(f"GNN Graph Structure: {explanation['video_name']}\n"
                     f"Node color = Anomaly score, Node size = Score magnitude",
                     fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # Adjacency matrix heatmap
        ax2 = fig.add_subplot(gs[1, 0])
        im = ax2.imshow(adj, cmap='YlOrRd', aspect='auto')
        ax2.set_xlabel('Segment Index', fontsize=10)
        ax2.set_ylabel('Segment Index', fontsize=10)
        ax2.set_title('Adjacency Matrix (DSM Output)', fontsize=11, fontweight='bold')
        plt.colorbar(im, ax=ax2, label='Connection Strength')
        
        # Degree distribution
        ax3 = fig.add_subplot(gs[1, 1])
        degrees = (adj > threshold).sum(axis=1)
        bars = ax3.bar(range(n_segments), degrees, color=node_colors, alpha=0.8)
        ax3.set_xlabel('Segment Index', fontsize=10)
        ax3.set_ylabel('Node Degree', fontsize=10)
        ax3.set_title('Segment Connectivity', fontsize=11, fontweight='bold')
        ax3.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved graph visualization to: {save_path}")
    
    def generate_text_explanation(self, explanation: Dict) -> str:
        """
        Generate human-readable text explanation.
        """
        lines = []
        lines.append("=" * 60)
        lines.append(f"EXPLAINABILITY REPORT: {explanation['video_name']}")
        lines.append("=" * 60)
        lines.append("")
        
        # Prediction summary
        lines.append("📊 PREDICTION SUMMARY")
        lines.append("-" * 40)
        lines.append(f"  Classification: {explanation['prediction']}")
        lines.append(f"  Max Anomaly Score: {explanation['max_score']:.4f}")
        lines.append(f"  Mean Anomaly Score: {explanation['mean_score']:.4f}")
        lines.append("")
        
        # Segment analysis
        seg_analysis = explanation['segment_analysis']
        lines.append("🔍 SEGMENT ANALYSIS")
        lines.append("-" * 40)
        lines.append(f"  Total Segments: {len(explanation['anomaly_scores'])}")
        
        if seg_analysis['anomalous_regions']:
            lines.append(f"  Anomalous Regions Detected: {len(seg_analysis['anomalous_regions'])}")
            for i, region in enumerate(seg_analysis['anomalous_regions'], 1):
                lines.append(f"    Region {i}: Segments {region['start']}-{region['end']} "
                           f"(Max: {region['max_score']:.3f})")
        else:
            lines.append("  No significant anomalous regions detected.")
        
        lines.append(f"\n  Top 5 Important Segments:")
        for idx, (seg, score) in enumerate(zip(seg_analysis['top_segments'], 
                                               seg_analysis['top_scores']), 1):
            lines.append(f"    {idx}. Segment {seg}: Score = {score:.4f}")
        lines.append("")
        
        # Graph analysis
        if 'graph_analysis' in explanation:
            graph = explanation['graph_analysis']
            lines.append("🌐 GRAPH RELATIONSHIP ANALYSIS")
            lines.append("-" * 40)
            lines.append(f"  Mean Node Degree: {graph['mean_degree']:.2f}")
            lines.append(f"  Graph Density: {graph['graph_density']:.4f}")
            lines.append(f"  Degree-Score Correlation: {graph['degree_score_correlation']:.4f}")
            
            if graph['high_connectivity_segments']:
                lines.append(f"  High-Connectivity Segments: {graph['high_connectivity_segments']}")
            lines.append("")
        
        # Interpretation
        lines.append("💡 INTERPRETATION")
        lines.append("-" * 40)
        
        if explanation['prediction'] == 'Abnormal':
            lines.append("  This video is classified as ABNORMAL because:")
            lines.append(f"  - The maximum anomaly score ({explanation['max_score']:.3f}) exceeds the threshold (0.5)")
            if seg_analysis['anomalous_regions']:
                total_anomalous = sum(r['end'] - r['start'] + 1 for r in seg_analysis['anomalous_regions'])
                lines.append(f"  - {total_anomalous} segments show anomalous behavior")
            lines.append("  - The model detected unusual patterns in the video segments")
        else:
            lines.append("  This video is classified as NORMAL because:")
            lines.append(f"  - All segment scores are below the threshold (0.5)")
            lines.append(f"  - The maximum score ({explanation['max_score']:.3f}) indicates normal behavior")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)


def generate_explainability_report(
    model_path: str,
    features_dir: str,
    output_dir: str,
    num_samples: int = 5
):
    """
    Generate explainability report for sample videos.
    
    Args:
        model_path: Path to trained model checkpoint
        features_dir: Directory containing video features
        output_dir: Output directory for reports
        num_samples: Number of sample videos to analyze
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # This is a placeholder - in real usage, load actual model and features
    print(f"Generating explainability report...")
    print(f"  Model: {model_path}")
    print(f"  Features: {features_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Samples: {num_samples}")


# Demo function for generating sample explanations
def generate_demo_explanation(output_dir: str):
    """Generate demo explanation visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    
    np.random.seed(42)
    n_segments = 32
    
    # Simulate anomaly scores for an abnormal video
    scores_abnormal = np.random.rand(n_segments) * 0.3
    scores_abnormal[12:18] = 0.6 + np.random.rand(6) * 0.35  # Anomalous region
    scores_abnormal[25:28] = 0.55 + np.random.rand(3) * 0.2  # Another anomalous region
    
    # Simulate adjacency matrix
    adj = np.random.rand(n_segments, n_segments)
    adj = (adj + adj.T) / 2  # Make symmetric
    adj = adj * 0.3 + np.eye(n_segments) * 0.7  # Add self-connections
    
    # Create mock explanation
    explanation_abnormal = {
        'video_name': 'Robbery_045_x264',
        'anomaly_scores': scores_abnormal.tolist(),
        'max_score': float(scores_abnormal.max()),
        'mean_score': float(scores_abnormal.mean()),
        'prediction': 'Abnormal',
        'segment_analysis': {
            'peaks': [14, 26],
            'anomalous_regions': [
                {'start': 12, 'end': 17, 'max_score': 0.92, 'mean_score': 0.78},
                {'start': 25, 'end': 27, 'max_score': 0.71, 'mean_score': 0.64}
            ],
            'top_segments': [14, 15, 13, 26, 16],
            'top_scores': sorted(scores_abnormal, reverse=True)[:5]
        },
        'adjacency_matrix': adj.tolist(),
        'graph_analysis': {
            'mean_degree': 8.5,
            'high_connectivity_segments': [12, 14, 15, 16],
            'degree_score_correlation': 0.67,
            'graph_density': 0.28
        }
    }
    
    # Create explainer-like object
    class DemoExplainer:
        def visualize_segment_importance(self, exp, save_path, figsize=(14, 6)):
            ModelExplainer.visualize_segment_importance(None, exp, save_path, figsize)
        
        def visualize_graph_relationships(self, exp, save_path, figsize=(12, 10)):
            ModelExplainer.visualize_graph_relationships(None, exp, save_path, figsize)
        
        def generate_text_explanation(self, exp):
            return ModelExplainer.generate_text_explanation(None, exp)
    
    explainer = DemoExplainer()
    
    # Generate visualizations
    explainer.visualize_segment_importance(
        explanation_abnormal,
        os.path.join(output_dir, 'segment_importance_abnormal.png')
    )
    
    explainer.visualize_graph_relationships(
        explanation_abnormal,
        os.path.join(output_dir, 'graph_relationships_abnormal.png')
    )
    
    # Generate text report
    text_report = explainer.generate_text_explanation(explanation_abnormal)
    with open(os.path.join(output_dir, 'explanation_report.txt'), 'w') as f:
        f.write(text_report)
    print(f"Saved text explanation to: {os.path.join(output_dir, 'explanation_report.txt')}")
    
    # Also generate for a normal video
    scores_normal = np.random.rand(n_segments) * 0.35
    explanation_normal = {
        'video_name': 'Normal_Videos_452_x264',
        'anomaly_scores': scores_normal.tolist(),
        'max_score': float(scores_normal.max()),
        'mean_score': float(scores_normal.mean()),
        'prediction': 'Normal',
        'segment_analysis': {
            'peaks': [],
            'anomalous_regions': [],
            'top_segments': list(np.argsort(scores_normal)[-5:][::-1]),
            'top_scores': sorted(scores_normal, reverse=True)[:5]
        },
        'adjacency_matrix': adj.tolist(),
        'graph_analysis': {
            'mean_degree': 7.2,
            'high_connectivity_segments': [],
            'degree_score_correlation': 0.12,
            'graph_density': 0.24
        }
    }
    
    explainer.visualize_segment_importance(
        explanation_normal,
        os.path.join(output_dir, 'segment_importance_normal.png')
    )
    
    # Save explanation data as JSON
    with open(os.path.join(output_dir, 'explanations.json'), 'w') as f:
        json.dump({
            'abnormal_video': explanation_abnormal,
            'normal_video': explanation_normal
        }, f, indent=2)
    
    print(f"\nDemo explanations generated in: {output_dir}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Explainability Reports')
    parser.add_argument('--model', type=str, help='Path to model checkpoint')
    parser.add_argument('--features', type=str, help='Path to features directory')
    parser.add_argument('--output', type=str, default='./explainability_output')
    parser.add_argument('--demo', action='store_true', help='Generate demo explanations')
    
    args = parser.parse_args()
    
    if args.demo:
        output_dir = os.path.join(project_root, 'experiments', 'results', 'explainability')
        generate_demo_explanation(output_dir)
    elif args.model and args.features:
        generate_explainability_report(args.model, args.features, args.output)
    else:
        print("Use --demo for demonstration or provide --model and --features paths.")
