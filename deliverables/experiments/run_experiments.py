"""
Comprehensive Experiment Runner for Video Anomaly Detection.

Runs experiments for all model configurations and generates results
for ablation studies and comparative analysis.
"""

import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    precision_score, recall_score, f1_score, accuracy_score
)

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from utils.config import Config, get_config, update_config
from utils.dataloader import get_dataloaders
from models.proposed_model import ProposedModel, build_model
from modules.mil_loss import MILLoss, get_loss_fn


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    description: str
    use_dsm: bool = True
    use_gnn: bool = True
    use_ra2r: bool = True
    num_segments: int = 32
    gnn_layers: int = 2
    epochs: int = 50
    
    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================================
# EXPERIMENT CONFIGURATIONS
# ============================================================================

EXPERIMENTS = {
    # Main comparative experiments
    'baseline_mil': ExperimentConfig(
        name='baseline_mil',
        description='Baseline MIL model without DSM, GNN, or RA2R',
        use_dsm=False, use_gnn=False, use_ra2r=False
    ),
    'mil_dsm': ExperimentConfig(
        name='mil_dsm',
        description='MIL + Dynamic Similarity Module',
        use_dsm=True, use_gnn=False, use_ra2r=False
    ),
    'mil_dsm_ra2r': ExperimentConfig(
        name='mil_dsm_ra2r',
        description='MIL + DSM + Relation-Aware Reasoning',
        use_dsm=True, use_gnn=False, use_ra2r=True
    ),
    'proposed_full': ExperimentConfig(
        name='proposed_full',
        description='Full Proposed Model: MIL + DSM + RA2R + GNN',
        use_dsm=True, use_gnn=True, use_ra2r=True
    ),
    
    # Ablation: Component removal
    'ablation_no_dsm': ExperimentConfig(
        name='ablation_no_dsm',
        description='Ablation: Without DSM',
        use_dsm=False, use_gnn=True, use_ra2r=True
    ),
    'ablation_no_ra2r': ExperimentConfig(
        name='ablation_no_ra2r',
        description='Ablation: Without RA2R',
        use_dsm=True, use_gnn=True, use_ra2r=False
    ),
    'ablation_no_gnn': ExperimentConfig(
        name='ablation_no_gnn',
        description='Ablation: Without GNN',
        use_dsm=True, use_gnn=False, use_ra2r=True
    ),
    
    # Ablation: Segment lengths
    'ablation_seg_16': ExperimentConfig(
        name='ablation_seg_16',
        description='Ablation: 16 segments',
        use_dsm=True, use_gnn=True, use_ra2r=True,
        num_segments=16
    ),
    'ablation_seg_64': ExperimentConfig(
        name='ablation_seg_64',
        description='Ablation: 64 segments',
        use_dsm=True, use_gnn=True, use_ra2r=True,
        num_segments=64
    ),
    
    # Ablation: GNN layers
    'ablation_gnn_1': ExperimentConfig(
        name='ablation_gnn_1',
        description='Ablation: 1 GNN layer',
        use_dsm=True, use_gnn=True, use_ra2r=True,
        gnn_layers=1
    ),
    'ablation_gnn_3': ExperimentConfig(
        name='ablation_gnn_3',
        description='Ablation: 3 GNN layers',
        use_dsm=True, use_gnn=True, use_ra2r=True,
        gnn_layers=3
    ),
    'ablation_gnn_4': ExperimentConfig(
        name='ablation_gnn_4',
        description='Ablation: 4 GNN layers',
        use_dsm=True, use_gnn=True, use_ra2r=True,
        gnn_layers=4
    ),
}


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def compute_all_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute comprehensive metrics for anomaly detection.
    
    Args:
        predictions: Anomaly scores (0-1)
        labels: Ground truth binary labels
        threshold: Threshold for binary classification
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # ROC-AUC
    try:
        metrics['roc_auc'] = roc_auc_score(labels, predictions)
    except ValueError:
        metrics['roc_auc'] = 0.5
    
    # PR-AUC
    precision_curve, recall_curve, _ = precision_recall_curve(labels, predictions)
    metrics['pr_auc'] = auc(recall_curve, precision_curve)
    
    # Binary predictions for classification metrics
    binary_preds = (predictions >= threshold).astype(int)
    
    # Precision, Recall, F1
    metrics['precision'] = precision_score(labels, binary_preds, zero_division=0)
    metrics['recall'] = recall_score(labels, binary_preds, zero_division=0)
    metrics['f1_score'] = f1_score(labels, binary_preds, zero_division=0)
    
    # Accuracy
    metrics['accuracy'] = accuracy_score(labels, binary_preds)
    
    return metrics


def train_model(
    model: nn.Module,
    train_loader,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epochs: int,
    exp_name: str,
    log_dir: str
) -> Tuple[nn.Module, List[Dict]]:
    """
    Train model and return training history.
    """
    writer = SummaryWriter(os.path.join(log_dir, exp_name))
    history = []
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            features = batch['features'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            output = model(features)
            scores = output['scores']
            
            loss, loss_dict = loss_fn(scores, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        history.append({'epoch': epoch + 1, 'loss': avg_loss})
        
        writer.add_scalar('Loss/train', avg_loss, epoch)
        
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch [{epoch + 1}/{epochs}] Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
    
    writer.close()
    return model, history


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    test_loader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Evaluate model and return predictions.
    """
    model.eval()
    
    all_scores = []
    all_labels = []
    video_predictions = {}
    
    for batch in test_loader:
        features = batch['features'].to(device)
        labels = batch['label']
        video_names = batch.get('video_names', [f'video_{i}' for i in range(len(labels))])
        
        output = model(features)
        scores = output['scores'].cpu().numpy()
        
        for i, name in enumerate(video_names):
            video_predictions[name] = scores[i]
            all_scores.append(scores[i].max())  # Max pooling for video-level
            all_labels.append(labels[i].item())
    
    return np.array(all_scores), np.array(all_labels), video_predictions


def run_single_experiment(
    exp_config: ExperimentConfig,
    base_config: Config,
    device: torch.device,
    results_dir: str
) -> Dict:
    """
    Run a single experiment configuration.
    """
    print(f"\n{'='*60}")
    print(f"Running: {exp_config.name}")
    print(f"Description: {exp_config.description}")
    print(f"{'='*60}")
    
    # Update config
    config = update_config(
        num_segments=exp_config.num_segments,
        gnn_layers=exp_config.gnn_layers,
        epochs=exp_config.epochs
    )
    
    set_seed(config.seed)
    
    # Get data loaders
    train_loader, test_loader = get_dataloaders(config)
    
    # Build model with specified configuration
    model = ProposedModel(
        feature_dim=config.feature_dim,
        hidden_dim=config.hidden_dim,
        output_dim=config.output_dim,
        num_segments=exp_config.num_segments,
        gnn_layers=exp_config.gnn_layers,
        gnn_heads=config.gnn_heads,
        ra2r_layers=config.ra2r_layers,
        ra2r_heads=config.ra2r_heads,
        dropout=config.dropout,
        use_dsm=exp_config.use_dsm,
        use_gnn=exp_config.use_gnn,
        use_ra2r=exp_config.use_ra2r
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Parameters: {trainable_params:,}")
    
    # Loss and optimizer
    loss_fn = get_loss_fn(
        loss_type=config.loss_type,
        topk=config.mil_topk,
        margin=config.mil_margin
    )
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Train
    log_dir = os.path.join(results_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    model, history = train_model(
        model, train_loader, loss_fn, optimizer,
        device, exp_config.epochs, exp_config.name, log_dir
    )
    
    # Evaluate
    predictions, labels, video_preds = evaluate_model(model, test_loader, device)
    
    # Compute metrics
    metrics = compute_all_metrics(predictions, labels)
    
    # Compile results
    result = {
        'experiment': exp_config.to_dict(),
        'metrics': metrics,
        'parameters': {
            'total': total_params,
            'trainable': trainable_params
        },
        'training_history': history,
        'timestamp': datetime.now().isoformat()
    }
    
    print(f"\n  Results:")
    print(f"    ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"    PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall: {metrics['recall']:.4f}")
    print(f"    F1-Score: {metrics['f1_score']:.4f}")
    
    # Save individual result
    result_path = os.path.join(results_dir, f'{exp_config.name}_results.json')
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    # Save model checkpoint
    checkpoint_path = os.path.join(results_dir, 'checkpoints', f'{exp_config.name}_model.pth')
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': exp_config.to_dict(),
        'metrics': metrics
    }, checkpoint_path)
    
    return result


def run_all_experiments(experiments: List[str] = None, epochs: int = 50):
    """
    Run all specified experiments.
    """
    # Setup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(project_root, 'experiments', 'results', f'run_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    config = get_config()
    
    # Select experiments
    if experiments is None:
        experiments = list(EXPERIMENTS.keys())
    
    all_results = {}
    
    for exp_name in experiments:
        if exp_name not in EXPERIMENTS:
            print(f"Warning: Unknown experiment '{exp_name}', skipping.")
            continue
        
        exp_config = EXPERIMENTS[exp_name]
        if epochs:
            exp_config.epochs = epochs
        
        try:
            result = run_single_experiment(exp_config, config, device, results_dir)
            all_results[exp_name] = result
        except Exception as e:
            print(f"Error in experiment '{exp_name}': {e}")
            import traceback
            traceback.print_exc()
    
    # Save combined results
    combined_path = os.path.join(results_dir, 'all_results.json')
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Generate summary table
    generate_results_table(all_results, results_dir)
    
    print(f"\n{'='*60}")
    print(f"All experiments completed!")
    print(f"Results saved to: {results_dir}")
    print(f"{'='*60}")
    
    return all_results


def generate_results_table(results: Dict, output_dir: str):
    """
    Generate a formatted results table.
    """
    table_lines = []
    table_lines.append("=" * 100)
    table_lines.append("EXPERIMENT RESULTS SUMMARY")
    table_lines.append("=" * 100)
    table_lines.append(f"{'Experiment':<25} {'ROC-AUC':>10} {'PR-AUC':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    table_lines.append("-" * 100)
    
    for name, result in results.items():
        metrics = result['metrics']
        table_lines.append(
            f"{name:<25} "
            f"{metrics['roc_auc']:>10.4f} "
            f"{metrics['pr_auc']:>10.4f} "
            f"{metrics['precision']:>10.4f} "
            f"{metrics['recall']:>10.4f} "
            f"{metrics['f1_score']:>10.4f}"
        )
    
    table_lines.append("=" * 100)
    
    table_text = "\n".join(table_lines)
    print("\n" + table_text)
    
    # Save to file
    table_path = os.path.join(output_dir, 'results_summary.txt')
    with open(table_path, 'w') as f:
        f.write(table_text)
    
    # Also save as CSV for easy import
    csv_path = os.path.join(output_dir, 'results_summary.csv')
    with open(csv_path, 'w') as f:
        f.write("Experiment,ROC-AUC,PR-AUC,Precision,Recall,F1-Score,Accuracy,Parameters\n")
        for name, result in results.items():
            metrics = result['metrics']
            params = result['parameters']['trainable']
            f.write(f"{name},{metrics['roc_auc']:.4f},{metrics['pr_auc']:.4f},"
                   f"{metrics['precision']:.4f},{metrics['recall']:.4f},"
                   f"{metrics['f1_score']:.4f},{metrics['accuracy']:.4f},{params}\n")


def main():
    parser = argparse.ArgumentParser(description='Run Video Anomaly Detection Experiments')
    parser.add_argument('--experiments', nargs='+', default=None,
                       help='List of experiments to run (default: all)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--list', action='store_true',
                       help='List available experiments')
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable Experiments:")
        print("-" * 60)
        for name, config in EXPERIMENTS.items():
            print(f"  {name:<25} - {config.description}")
        return
    
    run_all_experiments(args.experiments, args.epochs)


if __name__ == '__main__':
    main()
