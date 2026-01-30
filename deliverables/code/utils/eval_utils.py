"""
Evaluation utilities for Video Anomaly Detection.

Provides functions for computing evaluation metrics and visualizing results.
"""

import numpy as np
import os
from typing import List, Tuple, Optional
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc


def compute_auc(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    return_curve: bool = False
) -> Tuple[float, Optional[Tuple]]:
    """
    Compute Area Under the ROC Curve (AUC).
    
    Args:
        predictions: Anomaly scores, shape (N,) or flattened.
        ground_truth: Binary ground truth labels, same shape as predictions.
        return_curve: If True, also return the ROC curve points.
        
    Returns:
        auc_score: The AUC-ROC score.
        curve_data: Tuple of (fpr, tpr, thresholds) if return_curve is True.
    """
    # Flatten arrays if needed
    predictions = predictions.flatten()
    ground_truth = ground_truth.flatten()
    
    # Ensure same length
    assert len(predictions) == len(ground_truth), \
        f"Length mismatch: predictions={len(predictions)}, gt={len(ground_truth)}"
    
    # Compute AUC
    try:
        auc_score = roc_auc_score(ground_truth, predictions)
    except ValueError as e:
        print(f"Warning: Could not compute AUC ({e}). Returning 0.5")
        auc_score = 0.5
    
    if return_curve:
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(ground_truth, predictions)
        return auc_score, (fpr, tpr, thresholds)
    
    return auc_score, None


def compute_ap(
    predictions: np.ndarray,
    ground_truth: np.ndarray
) -> float:
    """
    Compute Average Precision (AP).
    
    Args:
        predictions: Anomaly scores.
        ground_truth: Binary ground truth labels.
        
    Returns:
        Average Precision score.
    """
    predictions = predictions.flatten()
    ground_truth = ground_truth.flatten()
    
    precision, recall, _ = precision_recall_curve(ground_truth, predictions)
    ap = auc(recall, precision)
    
    return ap


def normalize_scores(
    scores: np.ndarray,
    method: str = 'minmax'
) -> np.ndarray:
    """
    Normalize anomaly scores to [0, 1] range.
    
    Args:
        scores: Raw anomaly scores.
        method: Normalization method ('minmax', 'zscore', 'sigmoid').
        
    Returns:
        Normalized scores in [0, 1] range.
    """
    if method == 'minmax':
        min_val = scores.min()
        max_val = scores.max()
        if max_val - min_val > 1e-6:
            return (scores - min_val) / (max_val - min_val)
        else:
            return np.zeros_like(scores)
    
    elif method == 'zscore':
        mean = scores.mean()
        std = scores.std()
        if std > 1e-6:
            z_scores = (scores - mean) / std
            # Apply sigmoid to map to [0, 1]
            return 1 / (1 + np.exp(-z_scores))
        else:
            return np.zeros_like(scores)
    
    elif method == 'sigmoid':
        return 1 / (1 + np.exp(-scores))
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def interpolate_scores(
    scores: np.ndarray,
    target_length: int
) -> np.ndarray:
    """
    Interpolate anomaly scores to match target frame count.
    
    Used to align segment-level predictions with frame-level ground truth.
    
    Args:
        scores: Segment-level scores, shape (num_segments,).
        target_length: Number of frames in the video.
        
    Returns:
        Frame-level scores, shape (target_length,).
    """
    num_segments = len(scores)
    
    # Create interpolation indices
    indices = np.linspace(0, num_segments - 1, target_length)
    
    # Linear interpolation
    frame_scores = np.interp(
        np.arange(target_length),
        np.linspace(0, target_length - 1, num_segments),
        scores
    )
    
    return frame_scores


def smooth_scores(
    scores: np.ndarray,
    window_size: int = 5
) -> np.ndarray:
    """
    Apply temporal smoothing to anomaly scores.
    
    Uses a simple moving average filter.
    
    Args:
        scores: Raw scores.
        window_size: Size of smoothing window (must be odd).
        
    Returns:
        Smoothed scores.
    """
    if window_size % 2 == 0:
        window_size += 1
    
    kernel = np.ones(window_size) / window_size
    # Pad to handle edges
    padded = np.pad(scores, (window_size // 2, window_size // 2), mode='edge')
    smoothed = np.convolve(padded, kernel, mode='valid')
    
    return smoothed


def load_ground_truth(gt_path: str) -> dict:
    """
    Load ground truth annotations.
    
    Supports two formats:
    1. Video-level: "video_name label" per line
    2. Frame-level: "video_name frame1 frame2 ..." per line
    
    Args:
        gt_path: Path to ground truth file.
        
    Returns:
        Dictionary mapping video names to labels or frame indices.
    """
    gt_dict = {}
    
    with open(gt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                video_name = parts[0]
                # Check if video-level (single label) or frame-level
                if len(parts) == 2:
                    gt_dict[video_name] = int(parts[1])
                else:
                    # Frame-level annotations
                    gt_dict[video_name] = [int(x) for x in parts[1:]]
    
    return gt_dict


def plot_anomaly_curve(
    scores: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    video_name: str = "video",
    save_path: Optional[str] = None
) -> None:
    """
    Plot anomaly scores over time (optional).
    
    Args:
        scores: Anomaly scores for each frame/segment.
        ground_truth: Optional ground truth labels for comparison.
        video_name: Name of the video for the title.
        save_path: If provided, save the plot to this path.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available. Skipping plot.")
        return
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Plot scores
    frames = np.arange(len(scores))
    ax.plot(frames, scores, 'b-', label='Anomaly Score', linewidth=1.5)
    
    # Plot ground truth if available
    if ground_truth is not None:
        ax.fill_between(
            frames, 
            0, 
            ground_truth.max() if ground_truth.max() > 0 else 1,
            where=ground_truth > 0,
            alpha=0.3,
            color='red',
            label='Ground Truth'
        )
    
    ax.set_xlabel('Frame')
    ax.set_ylabel('Anomaly Score')
    ax.set_title(f'Anomaly Detection: {video_name}')
    ax.legend()
    ax.set_ylim([0, 1.1])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def evaluate_all_videos(
    predictions: dict,
    ground_truth: dict,
    frame_counts: Optional[dict] = None
) -> dict:
    """
    Evaluate predictions across all test videos.
    
    Args:
        predictions: Dict mapping video_name -> segment scores.
        ground_truth: Dict mapping video_name -> frame-level GT or video-level label.
        frame_counts: Dict mapping video_name -> number of frames (for interpolation).
        
    Returns:
        Dictionary with evaluation metrics.
    """
    all_scores = []
    all_gt = []
    
    for video_name, scores in predictions.items():
        if video_name not in ground_truth:
            continue
        
        gt = ground_truth[video_name]
        
        # Handle video-level vs frame-level GT
        if isinstance(gt, int):
            # Video-level: expand to segment-level
            video_gt = np.full(len(scores), gt)
        else:
            # Frame-level: interpolate scores to match
            target_len = len(gt) if isinstance(gt, list) else len(gt)
            if frame_counts and video_name in frame_counts:
                target_len = frame_counts[video_name]
            scores = interpolate_scores(scores, target_len)
            video_gt = np.array(gt) if isinstance(gt, list) else gt
        
        all_scores.extend(scores)
        all_gt.extend(video_gt if isinstance(video_gt, np.ndarray) else [video_gt])
    
    all_scores = np.array(all_scores)
    all_gt = np.array(all_gt)
    
    # Compute metrics
    auc_score, curve = compute_auc(all_scores, all_gt, return_curve=True)
    ap_score = compute_ap(all_scores, all_gt)
    
    return {
        'AUC': auc_score,
        'AP': ap_score,
        'num_frames': len(all_scores),
        'num_positive': int(all_gt.sum()),
        'num_negative': int((1 - all_gt).sum())
    }


if __name__ == '__main__':
    # Test evaluation utilities
    np.random.seed(42)
    
    # Generate dummy predictions and ground truth
    predictions = np.random.rand(100)
    ground_truth = (np.random.rand(100) > 0.7).astype(int)
    
    auc_score, _ = compute_auc(predictions, ground_truth)
    print(f"Test AUC: {auc_score:.4f}")
    
    normalized = normalize_scores(predictions)
    print(f"Normalized range: [{normalized.min():.4f}, {normalized.max():.4f}]")
