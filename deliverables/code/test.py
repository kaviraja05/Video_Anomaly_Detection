"""
Testing/Evaluation script for Video Anomaly Detection.

Implements the complete evaluation pipeline with:
- Model loading from checkpoint
- Inference on test data
- AUC-ROC computation
- Results saving
"""

import os
import sys
import argparse
import json
import numpy as np
from datetime import datetime
from typing import Dict, List

import torch
import torch.nn as nn
from tqdm import tqdm

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from utils.config import get_config, update_config
from utils.dataloader import get_dataloaders, load_video_list
from utils.eval_utils import (
    compute_auc, compute_ap, normalize_scores,
    interpolate_scores, smooth_scores, plot_anomaly_curve
)
from models.proposed_model import ProposedModel, build_model


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test Video Anomaly Detection Model')
    
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (default: best_model.pth)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--save_scores', action='store_true',
                        help='Save per-video anomaly scores')
    parser.add_argument('--plot', action='store_true',
                        help='Generate anomaly score plots')
    parser.add_argument('--normalize', type=str, default='minmax',
                        choices=['minmax', 'zscore', 'sigmoid', 'none'],
                        help='Score normalization method')
    parser.add_argument('--smooth', type=int, default=0,
                        help='Smoothing window size (0 for no smoothing)')
    
    return parser.parse_args()


def load_model(checkpoint_path: str, config, device: torch.device) -> nn.Module:
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file.
        config: Configuration object.
        device: Device to load model on.
        
    Returns:
        Loaded model.
    """
    # Build model
    model = build_model(config)
    model = model.to(device)
    
    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    if 'metrics' in checkpoint:
        print(f"  Training loss: {checkpoint['metrics'].get('loss', 'N/A'):.4f}")
    
    return model


@torch.no_grad()
def inference(
    model: nn.Module,
    test_loader,
    device: torch.device,
    normalize: str = 'minmax',
    smooth_window: int = 0
) -> Dict[str, np.ndarray]:
    """
    Run inference on test set.
    
    Args:
        model: Trained model.
        test_loader: Test data loader.
        device: Device.
        normalize: Normalization method.
        smooth_window: Smoothing window size.
        
    Returns:
        Dictionary mapping video names to anomaly scores.
    """
    model.eval()
    
    predictions = {}
    labels_dict = {}
    
    for batch in tqdm(test_loader, desc="Running inference"):
        features = batch['features'].to(device)  # (1, T, D)
        labels = batch['label']  # (1,)
        video_names = batch['video_names']  # List of 1 name
        
        # Forward pass
        output = model(features)
        scores = output['scores'].cpu().numpy()  # (1, T)
        
        for i, video_name in enumerate(video_names):
            video_scores = scores[i]  # (T,)
            
            # Apply smoothing if requested
            if smooth_window > 0:
                video_scores = smooth_scores(video_scores, smooth_window)
            
            predictions[video_name] = video_scores
            labels_dict[video_name] = labels[i].item()
    
    # Normalize all scores together for consistent comparison
    if normalize != 'none':
        all_scores = np.concatenate([s for s in predictions.values()])
        
        # Get normalization parameters
        min_val, max_val = all_scores.min(), all_scores.max()
        
        # Apply to each video
        for video_name in predictions:
            scores = predictions[video_name]
            if normalize == 'minmax':
                if max_val - min_val > 1e-6:
                    predictions[video_name] = (scores - min_val) / (max_val - min_val)
            elif normalize == 'zscore':
                mean, std = all_scores.mean(), all_scores.std()
                if std > 1e-6:
                    z = (scores - mean) / std
                    predictions[video_name] = 1 / (1 + np.exp(-z))
            elif normalize == 'sigmoid':
                predictions[video_name] = 1 / (1 + np.exp(-scores))
    
    return predictions, labels_dict


def compute_metrics(
    predictions: Dict[str, np.ndarray],
    labels_dict: Dict[str, int]
) -> Dict[str, float]:
    """
    Compute evaluation metrics.
    
    Args:
        predictions: Dictionary of video_name -> segment scores.
        labels_dict: Dictionary of video_name -> label (0 or 1).
        
    Returns:
        Dictionary of metric_name -> value.
    """
    metrics = {}
    
    # =========================================================================
    # Video-level AUC
    # =========================================================================
    # For video-level: use max score per video as the prediction
    video_scores = []
    video_labels = []
    
    for video_name, scores in predictions.items():
        if video_name in labels_dict:
            video_scores.append(scores.max())  # Max pooling
            video_labels.append(labels_dict[video_name])
    
    video_scores = np.array(video_scores)
    video_labels = np.array(video_labels)
    
    if len(np.unique(video_labels)) > 1:
        video_auc, _ = compute_auc(video_scores, video_labels)
        video_ap = compute_ap(video_scores, video_labels)
        metrics['video_auc'] = video_auc
        metrics['video_ap'] = video_ap
    else:
        metrics['video_auc'] = 0.5
        metrics['video_ap'] = 0.5
    
    # =========================================================================
    # Segment-level statistics
    # =========================================================================
    all_scores = np.concatenate([s for s in predictions.values()])
    metrics['mean_score'] = float(all_scores.mean())
    metrics['std_score'] = float(all_scores.std())
    metrics['min_score'] = float(all_scores.min())
    metrics['max_score'] = float(all_scores.max())
    
    # Normal vs Abnormal video mean scores
    normal_scores = []
    abnormal_scores = []
    for video_name, scores in predictions.items():
        if video_name in labels_dict:
            if labels_dict[video_name] == 0:
                normal_scores.append(scores.mean())
            else:
                abnormal_scores.append(scores.mean())
    
    if normal_scores:
        metrics['mean_normal_score'] = float(np.mean(normal_scores))
    if abnormal_scores:
        metrics['mean_abnormal_score'] = float(np.mean(abnormal_scores))
    
    # =========================================================================
    # Number of videos
    # =========================================================================
    metrics['num_videos'] = len(predictions)
    metrics['num_normal'] = sum(1 for v in labels_dict.values() if v == 0)
    metrics['num_abnormal'] = sum(1 for v in labels_dict.values() if v == 1)
    
    return metrics


def save_results(
    metrics: Dict[str, float],
    predictions: Dict[str, np.ndarray],
    labels_dict: Dict[str, int],
    config,
    save_scores: bool = True
):
    """
    Save evaluation results.
    
    Args:
        metrics: Evaluation metrics.
        predictions: Per-video predictions.
        labels_dict: Per-video labels.
        config: Configuration object.
        save_scores: Whether to save per-video scores.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save metrics
    metrics_path = os.path.join(config.results_dir, f'metrics_{timestamp}.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")
    
    # Save per-video scores
    if save_scores:
        scores_path = os.path.join(config.results_dir, f'scores_{timestamp}.npz')
        np.savez(scores_path, **predictions)
        print(f"Scores saved to: {scores_path}")
        
        # Save labels
        labels_path = os.path.join(config.results_dir, f'labels_{timestamp}.json')
        with open(labels_path, 'w') as f:
            json.dump(labels_dict, f, indent=2)


def generate_plots(
    predictions: Dict[str, np.ndarray],
    labels_dict: Dict[str, int],
    config,
    max_plots: int = 10
):
    """
    Generate anomaly score plots for visualization.
    
    Args:
        predictions: Per-video predictions.
        labels_dict: Per-video labels.
        config: Configuration object.
        max_plots: Maximum number of plots to generate.
    """
    plot_dir = os.path.join(config.results_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    # Select representative videos
    normal_videos = [v for v, l in labels_dict.items() if l == 0]
    abnormal_videos = [v for v, l in labels_dict.items() if l == 1]
    
    # Plot some of each type
    to_plot = abnormal_videos[:max_plots // 2] + normal_videos[:max_plots // 2]
    
    for video_name in to_plot:
        if video_name in predictions:
            scores = predictions[video_name]
            label = labels_dict.get(video_name, -1)
            label_str = "Abnormal" if label == 1 else "Normal"
            
            save_path = os.path.join(plot_dir, f'{video_name}_scores.png')
            plot_anomaly_curve(
                scores,
                video_name=f"{video_name} ({label_str})",
                save_path=save_path
            )
    
    print(f"Plots saved to: {plot_dir}")


def test(config, args):
    """
    Main testing function.
    
    Args:
        config: Configuration object.
        args: Command line arguments.
    """
    print("=" * 60)
    print("Video Anomaly Detection Evaluation")
    print("=" * 60)
    
    # Set device
    device = torch.device(
        args.device if args.device else 
        (config.device if torch.cuda.is_available() else 'cpu')
    )
    print(f"Using device: {device}")
    
    # Determine checkpoint path
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = os.path.join(config.checkpoint_dir, 'best_model.pth')
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(config.checkpoint_dir, 'latest_model.pth')
    
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load model
    model = load_model(checkpoint_path, config, device)
    
    # Create test loader
    print("\nLoading test data...")
    _, test_loader = get_dataloaders(config)
    print(f"  Test samples: {len(test_loader)}")
    
    # Run inference
    print("\nRunning inference...")
    predictions, labels_dict = inference(
        model, test_loader, device,
        normalize=args.normalize,
        smooth_window=args.smooth
    )
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(predictions, labels_dict)
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Video-level AUC: {metrics.get('video_auc', 0.0):.4f}")
    print(f"  Video-level AP:  {metrics.get('video_ap', 0.0):.4f}")
    print(f"  Mean Normal Score:   {metrics.get('mean_normal_score', 0.0):.4f}")
    print(f"  Mean Abnormal Score: {metrics.get('mean_abnormal_score', 0.0):.4f}")
    print(f"  Number of Videos: {metrics.get('num_videos', 0)}")
    print(f"    Normal: {metrics.get('num_normal', 0)}")
    print(f"    Abnormal: {metrics.get('num_abnormal', 0)}")
    
    # Save results
    save_results(
        metrics, predictions, labels_dict, config,
        save_scores=args.save_scores
    )
    
    # Generate plots if requested
    if args.plot:
        print("\nGenerating plots...")
        generate_plots(predictions, labels_dict, config)
    
    print("\n" + "=" * 60)
    print("Evaluation completed!")
    print("=" * 60)
    
    return metrics


def main():
    """Main entry point."""
    args = parse_args()
    config = get_config()
    
    if args.device:
        update_config(device=args.device)
    
    metrics = test(config, args)
    return metrics


if __name__ == '__main__':
    main()
