"""
Configuration module for Video Anomaly Detection.

Provides centralized configuration with all hyperparameters and paths
accessible throughout the project.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:
    """
    Centralized configuration for Video Anomaly Detection project.
    
    Attributes:
        Feature dimensions, model hyperparameters, training settings,
        and file paths are all defined here.
    """
    
    # ==========================================================================
    # Feature Dimensions
    # ==========================================================================
    feature_dim: int = 2048          # I3D feature dimension
    hidden_dim: int = 512            # Hidden layer dimension
    output_dim: int = 128            # Output embedding dimension
    num_segments: int = 32           # Number of segments per video
    
    # ==========================================================================
    # GNN Parameters
    # ==========================================================================
    gnn_layers: int = 2              # Number of GNN layers
    gnn_heads: int = 4               # Number of attention heads in GNN
    adjacency_threshold: float = 0.5 # Threshold for adjacency matrix
    
    # ==========================================================================
    # DSM (Dynamic Similarity Module) Parameters
    # ==========================================================================
    dsm_hidden_dim: int = 256        # DSM hidden dimension
    dsm_temperature: float = 0.1     # Temperature for softmax in DSM
    
    # ==========================================================================
    # RA2R (Relation-Aware Reasoning) Parameters
    # ==========================================================================
    ra2r_layers: int = 2             # Number of RA2R layers
    ra2r_heads: int = 4              # Attention heads in RA2R
    
    # ==========================================================================
    # Training Parameters
    # ==========================================================================
    batch_size: int = 30             # Batch size (normal + abnormal pairs)
    learning_rate: float = 1e-4      # Learning rate
    weight_decay: float = 1e-5       # Weight decay for regularization
    epochs: int = 100                # Number of training epochs
    dropout: float = 0.6             # Dropout rate
    
    # ==========================================================================
    # MIL Loss Parameters
    # ==========================================================================
    loss_type: str = 'mil'           # Loss type: 'mil', 'contrastive', 'focal'
    mil_topk: int = 3                # Top-k instances for MIL
    mil_margin: float = 1.0          # Margin for ranking loss
    smoothness_weight: float = 8e-5  # Smoothness regularization weight
    sparsity_weight: float = 8e-5    # Sparsity regularization weight
    
    # ==========================================================================
    # Paths
    # ==========================================================================
    project_root: str = field(default_factory=lambda: os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    ))
    
    @property
    def data_dir(self) -> str:
        return os.path.join(self.project_root, 'data')
    
    @property
    def feature_dir(self) -> str:
        return os.path.join(self.data_dir, 'i3d_features')
    
    @property
    def train_feature_dir(self) -> str:
        return os.path.join(self.feature_dir, 'train')
    
    @property
    def test_feature_dir(self) -> str:
        return os.path.join(self.feature_dir, 'test')
    
    @property
    def splits_dir(self) -> str:
        return os.path.join(self.data_dir, 'splits')
    
    @property
    def train_split_path(self) -> str:
        return os.path.join(self.splits_dir, 'train_split.txt')
    
    @property
    def test_split_path(self) -> str:
        return os.path.join(self.splits_dir, 'test_split.txt')
    
    @property
    def gt_path(self) -> str:
        return os.path.join(self.splits_dir, 'gt.txt')
    
    @property
    def checkpoint_dir(self) -> str:
        return os.path.join(self.project_root, 'experiments', 'checkpoints')
    
    @property
    def log_dir(self) -> str:
        return os.path.join(self.project_root, 'experiments', 'logs')
    
    @property
    def results_dir(self) -> str:
        return os.path.join(self.project_root, 'experiments', 'results')
    
    # ==========================================================================
    # Device Configuration
    # ==========================================================================
    device: str = 'cuda'             # Device: 'cuda' or 'cpu'
    seed: int = 42                   # Random seed for reproducibility
    
    # ==========================================================================
    # Logging
    # ==========================================================================
    log_interval: int = 10           # Log every N batches
    save_interval: int = 5           # Save checkpoint every N epochs
    
    def __post_init__(self):
        """Create necessary directories if they don't exist."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for logging."""
        return {
            'feature_dim': self.feature_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'num_segments': self.num_segments,
            'gnn_layers': self.gnn_layers,
            'gnn_heads': self.gnn_heads,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'dropout': self.dropout,
            'mil_topk': self.mil_topk,
            'mil_margin': self.mil_margin,
        }


# Global config instance for easy access
config = Config()


def get_config() -> Config:
    """Get the global configuration instance."""
    return config


def update_config(**kwargs) -> Config:
    """
    Update configuration with new values.
    
    Args:
        **kwargs: Configuration parameters to update.
        
    Returns:
        Updated Config instance.
    """
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown config parameter: {key}")
    return config
