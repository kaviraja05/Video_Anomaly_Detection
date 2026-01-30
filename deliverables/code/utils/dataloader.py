"""
Dataset loading utilities for Video Anomaly Detection.

Provides PyTorch Dataset and DataLoader implementations for loading
I3D features with proper segmentation and label alignment.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict, Optional


class AnomalyDataset(Dataset):
    """
    PyTorch Dataset for loading video anomaly detection data.
    
    Loads I3D features from .npy files and segments them into fixed-length
    sequences for batch processing.
    
    Attributes:
        feature_dir: Directory containing I3D feature files.
        video_list: List of video names to load.
        num_segments: Number of segments to divide each video into.
        is_train: Whether this is training data.
    """
    
    def __init__(
        self,
        feature_dir: str,
        video_list: List[str],
        num_segments: int = 32,
        is_train: bool = True
    ):
        """
        Initialize the dataset.
        
        Args:
            feature_dir: Path to directory with .npy feature files.
            video_list: List of video names (without .npy extension).
            num_segments: Number of temporal segments (default 32).
            is_train: Flag for training mode.
        """
        self.feature_dir = feature_dir
        self.video_list = video_list
        self.num_segments = num_segments
        self.is_train = is_train
        
    def __len__(self) -> int:
        return len(self.video_list)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single video's features and label.
        
        Args:
            idx: Index of the video.
            
        Returns:
            Dictionary containing:
                - features: Tensor of shape (num_segments, feature_dim)
                - label: 0 for normal, 1 for abnormal
                - video_name: Name of the video
        """
        video_name = self.video_list[idx]
        
        # Construct feature file path
        feature_path = os.path.join(self.feature_dir, f"{video_name}.npy")
        
        # Load I3D features
        features = np.load(feature_path)  # Shape: (T, D) where T=frames, D=2048
        
        # Segment features to fixed length
        features = self._segment_features(features)
        
        # Determine label from video name (Normal = 0, Abnormal = 1)
        label = 0 if 'Normal' in video_name else 1
        
        return {
            'features': torch.tensor(features, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long),
            'video_name': video_name
        }
    
    def _segment_features(self, features: np.ndarray) -> np.ndarray:
        """
        Divide video features into fixed number of segments.
        
        Handles both 2D (T, D) and 3D (T, snippets, D) I3D feature formats.
        Uses uniform sampling to select num_segments frames from the video.
        If video is shorter than num_segments, pad with zeros.
        
        Args:
            features: Raw features of shape (T, D) or (T, snippets, D).
            
        Returns:
            Segmented features of shape (num_segments, D).
        """
        # Handle 3D features (T, snippets, D) by averaging across snippets
        if len(features.shape) == 3:
            features = features.mean(axis=1)  # Average across snippets -> (T, D)
        
        T, D = features.shape
        
        if T >= self.num_segments:
            # Sample uniformly from the video
            indices = np.linspace(0, T - 1, self.num_segments, dtype=np.int32)
            return features[indices]
        else:
            # Pad with zeros if video is too short
            padded = np.zeros((self.num_segments, D), dtype=features.dtype)
            padded[:T] = features
            return padded


class TrainBatchSampler:
    """
    Custom batch sampler for training that ensures each batch contains
    both normal and abnormal videos (required for MIL loss).
    
    Each batch contains batch_size // 2 normal and batch_size // 2 abnormal videos.
    """
    
    def __init__(
        self,
        normal_indices: List[int],
        abnormal_indices: List[int],
        batch_size: int = 30
    ):
        """
        Initialize the batch sampler.
        
        Args:
            normal_indices: Indices of normal videos in the dataset.
            abnormal_indices: Indices of abnormal videos in the dataset.
            batch_size: Total batch size (must be even).
        """
        self.normal_indices = normal_indices
        self.abnormal_indices = abnormal_indices
        self.batch_size = batch_size
        self.half_batch = batch_size // 2
        
    def __iter__(self):
        # Shuffle indices for each epoch
        normal_perm = np.random.permutation(self.normal_indices)
        abnormal_perm = np.random.permutation(self.abnormal_indices)
        
        # Determine number of batches
        n_batches = min(
            len(normal_perm) // self.half_batch,
            len(abnormal_perm) // self.half_batch
        )
        
        for i in range(n_batches):
            normal_batch = normal_perm[i * self.half_batch:(i + 1) * self.half_batch]
            abnormal_batch = abnormal_perm[i * self.half_batch:(i + 1) * self.half_batch]
            
            # Yield combined batch
            batch = list(normal_batch) + list(abnormal_batch)
            yield batch
    
    def __len__(self):
        return min(
            len(self.normal_indices) // self.half_batch,
            len(self.abnormal_indices) // self.half_batch
        )


def load_video_list(split_path: str) -> List[str]:
    """
    Load list of video names from split file.
    
    Args:
        split_path: Path to split file (train_split.txt or test_split.txt).
        
    Returns:
        List of video names.
    """
    with open(split_path, 'r') as f:
        video_list = [line.strip() for line in f if line.strip()]
    return video_list


def create_split_files(config) -> None:
    """
    Create train/test split files from existing feature files.
    
    This function scans the feature directories and creates split files
    listing all available videos.
    
    Args:
        config: Configuration object with paths.
    """
    # Get train videos
    train_videos = []
    if os.path.exists(config.train_feature_dir):
        for f in os.listdir(config.train_feature_dir):
            if f.endswith('.npy'):
                video_name = f[:-4]  # Remove .npy extension
                train_videos.append(video_name)
    
    # Get test videos  
    test_videos = []
    if os.path.exists(config.test_feature_dir):
        for f in os.listdir(config.test_feature_dir):
            if f.endswith('.npy'):
                video_name = f[:-4]  # Remove .npy extension
                test_videos.append(video_name)
    
    # Write train split
    with open(config.train_split_path, 'w') as f:
        for video in sorted(train_videos):
            f.write(f"{video}\n")
    
    # Write test split
    with open(config.test_split_path, 'w') as f:
        for video in sorted(test_videos):
            f.write(f"{video}\n")
    
    # Create ground truth file (video-level labels)
    # For UCF-Crime style: abnormal videos have anomaly category prefix
    gt_lines = []
    for video in sorted(test_videos):
        label = 0 if 'Normal' in video else 1
        gt_lines.append(f"{video} {label}")
    
    with open(config.gt_path, 'w') as f:
        for line in gt_lines:
            f.write(f"{line}\n")
    
    print(f"Created split files:")
    print(f"  Train videos: {len(train_videos)}")
    print(f"  Test videos: {len(test_videos)}")


def get_dataloaders(config) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and testing DataLoaders.
    
    Args:
        config: Configuration object with all parameters.
        
    Returns:
        Tuple of (train_loader, test_loader).
    """
    # Ensure split files exist
    if not os.path.exists(config.train_split_path) or \
       os.path.getsize(config.train_split_path) == 0:
        print("Split files not found. Creating from feature directories...")
        create_split_files(config)
    
    # Load video lists
    train_videos = load_video_list(config.train_split_path)
    test_videos = load_video_list(config.test_split_path)
    
    # Create datasets
    train_dataset = AnomalyDataset(
        feature_dir=config.train_feature_dir,
        video_list=train_videos,
        num_segments=config.num_segments,
        is_train=True
    )
    
    test_dataset = AnomalyDataset(
        feature_dir=config.test_feature_dir,
        video_list=test_videos,
        num_segments=config.num_segments,
        is_train=False
    )
    
    # For training: use custom batch sampler for balanced batches
    normal_indices = [i for i, v in enumerate(train_videos) if 'Normal' in v]
    abnormal_indices = [i for i, v in enumerate(train_videos) if 'Normal' not in v]
    
    batch_sampler = TrainBatchSampler(
        normal_indices=normal_indices,
        abnormal_indices=abnormal_indices,
        batch_size=config.batch_size
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # For testing: sequential loading
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Process one video at a time for evaluation
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, test_loader


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching video data.
    
    Args:
        batch: List of sample dictionaries.
        
    Returns:
        Batched dictionary with stacked tensors.
    """
    features = torch.stack([item['features'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    video_names = [item['video_name'] for item in batch]
    
    return {
        'features': features,
        'label': labels,
        'video_names': video_names
    }


if __name__ == '__main__':
    # Test the dataloader
    from config import get_config
    
    config = get_config()
    train_loader, test_loader = get_dataloaders(config)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Test samples: {len(test_loader)}")
    
    # Test loading a batch
    for batch in train_loader:
        print(f"Batch features shape: {batch['features'].shape}")
        print(f"Batch labels: {batch['label']}")
        break
