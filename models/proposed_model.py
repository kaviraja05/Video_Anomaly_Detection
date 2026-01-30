"""
Proposed Model for Video Anomaly Detection.

Combines I3D feature embedding, Dynamic Similarity Module (DSM),
Graph Neural Networks, and Relation-Aware Reasoning (RA2R)
for anomaly detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base_model import FeatureEmbedding, TemporalModule, AnomalyScorer
from modules.gnn_layer import GNNBlock, compute_adjacency
from modules.dsm import DynamicSimilarityModule, TemporalDSM
from modules.ra2r import RelationAwareReasoning


class ProposedModel(nn.Module):
    """
    Full model combining I3D features + GNN + DSM + RA2R for anomaly detection.
    
    Architecture:
    1. Feature Embedding: Project I3D (2048) to compact representation
    2. Temporal Module: Capture local temporal patterns
    3. DSM: Learn dynamic similarity/adjacency between segments
    4. GNN: Message passing on the learned graph
    5. RA2R: Relation-aware reasoning for high-order dependencies
    6. Scorer: Per-segment anomaly scores
    """
    
    def __init__(
        self,
        feature_dim: int = 2048,
        hidden_dim: int = 512,
        output_dim: int = 128,
        num_segments: int = 32,
        gnn_layers: int = 2,
        gnn_heads: int = 4,
        ra2r_layers: int = 2,
        ra2r_heads: int = 4,
        dropout: float = 0.6,
        use_dsm: bool = True,
        use_gnn: bool = True,
        use_ra2r: bool = True
    ):
        """
        Initialize the proposed model.
        
        Args:
            feature_dim: I3D feature dimension (2048).
            hidden_dim: Hidden layer dimension.
            output_dim: Embedding/output dimension.
            num_segments: Number of video segments.
            gnn_layers: Number of GNN layers.
            gnn_heads: Attention heads in GNN.
            ra2r_layers: Number of RA2R layers.
            ra2r_heads: Attention heads in RA2R.
            dropout: Dropout rate.
            use_dsm: Whether to use Dynamic Similarity Module.
            use_gnn: Whether to use GNN.
            use_ra2r: Whether to use Relation-Aware Reasoning.
        """
        super().__init__()
        
        self.use_dsm = use_dsm
        self.use_gnn = use_gnn
        self.use_ra2r = use_ra2r
        self.num_segments = num_segments
        
        # =====================================================================
        # 1. Feature Embedding
        # =====================================================================
        self.embedding = FeatureEmbedding(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout
        )
        
        # =====================================================================
        # 2. Temporal Module
        # =====================================================================
        self.temporal = TemporalModule(
            input_dim=output_dim,
            hidden_dim=output_dim,
            dropout=dropout / 2
        )
        
        # =====================================================================
        # 3. Dynamic Similarity Module (DSM)
        # =====================================================================
        if use_dsm:
            self.dsm = TemporalDSM(
                input_dim=output_dim,
                hidden_dim=output_dim * 2,
                max_segments=num_segments,
                num_heads=gnn_heads,
                dropout=dropout / 2
            )
        
        # =====================================================================
        # 4. Graph Neural Network
        # =====================================================================
        if use_gnn:
            self.gnn = GNNBlock(
                input_dim=output_dim,
                hidden_dim=output_dim,
                output_dim=output_dim,
                num_layers=gnn_layers,
                num_heads=gnn_heads,
                dropout=dropout / 2
            )
        
        # =====================================================================
        # 5. Relation-Aware Reasoning (RA2R)
        # =====================================================================
        if use_ra2r:
            self.ra2r = RelationAwareReasoning(
                input_dim=output_dim,
                relation_dim=output_dim // 2,
                num_heads=ra2r_heads,
                num_layers=ra2r_layers,
                dropout=dropout / 2
            )
        
        # =====================================================================
        # 6. Anomaly Scorer
        # =====================================================================
        self.scorer = AnomalyScorer(
            input_dim=output_dim,
            hidden_dim=output_dim // 2,
            dropout=dropout / 2
        )
        
        # Optional: attention pooling for video-level features
        self.attention_pool = nn.Sequential(
            nn.Linear(output_dim, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the full model.
        
        Args:
            x: I3D features of shape (B, T, feature_dim).
            return_features: If True, also return intermediate features.
            
        Returns:
            Dictionary containing:
                - scores: Anomaly scores of shape (B, T)
                - features: Final segment features (optional)
                - adj: Adjacency matrix (optional, if using DSM)
        """
        B, T, D = x.shape
        output = {}
        
        # 1. Feature embedding
        x = self.embedding(x)  # (B, T, output_dim)
        
        # 2. Temporal modeling
        x = self.temporal(x)  # (B, T, output_dim)
        
        # 3. Dynamic Similarity Module
        adj = None
        if self.use_dsm:
            adj, _ = self.dsm(x)  # (B, T, T)
            output['adj'] = adj
        else:
            # Use cosine similarity as fallback
            adj = compute_adjacency(x, method='cosine', threshold=0.5)
        
        # 4. Graph Neural Network
        if self.use_gnn:
            x = self.gnn(x, adj)  # (B, T, output_dim)
        
        # 5. Relation-Aware Reasoning
        if self.use_ra2r:
            x, _ = self.ra2r(x, adj)  # (B, T, output_dim)
        
        # 6. Anomaly scoring
        scores = self.scorer(x)  # (B, T)
        output['scores'] = scores
        
        if return_features:
            output['features'] = x
            
            # Compute video-level representation using attention pooling
            attn_weights = self.attention_pool(x)  # (B, T, 1)
            video_features = (x * attn_weights).sum(dim=1)  # (B, output_dim)
            output['video_features'] = video_features
        
        return output
    
    def get_segment_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get per-segment anomaly scores (convenience method).
        
        Args:
            x: I3D features of shape (B, T, feature_dim).
            
        Returns:
            Anomaly scores of shape (B, T).
        """
        output = self.forward(x)
        return output['scores']


class LightweightModel(nn.Module):
    """
    Lightweight version of the model without RA2R.
    
    Faster training and inference while maintaining competitive performance.
    """
    
    def __init__(
        self,
        feature_dim: int = 2048,
        hidden_dim: int = 256,
        output_dim: int = 64,
        gnn_layers: int = 2,
        gnn_heads: int = 4,
        dropout: float = 0.5
    ):
        """Initialize the lightweight model."""
        super().__init__()
        
        # Simpler embedding
        self.embedding = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # DSM
        self.dsm = DynamicSimilarityModule(
            input_dim=output_dim,
            hidden_dim=output_dim,
            num_heads=gnn_heads,
            dropout=dropout / 2
        )
        
        # Single GNN block
        self.gnn = GNNBlock(
            input_dim=output_dim,
            hidden_dim=output_dim,
            output_dim=output_dim,
            num_layers=gnn_layers,
            num_heads=gnn_heads,
            dropout=dropout / 2
        )
        
        # Scorer
        self.scorer = nn.Sequential(
            nn.Linear(output_dim, output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(output_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        # Embed
        x = self.embedding(x)  # (B, T, output_dim)
        
        # DSM
        adj, _ = self.dsm(x)
        
        # GNN
        x = self.gnn(x, adj)
        
        # Score
        scores = self.scorer(x).squeeze(-1)  # (B, T)
        
        return {'scores': scores, 'adj': adj}


def build_model(config) -> nn.Module:
    """
    Build model from configuration.
    
    Args:
        config: Configuration object with model parameters.
        
    Returns:
        Initialized model.
    """
    model = ProposedModel(
        feature_dim=config.feature_dim,
        hidden_dim=config.hidden_dim,
        output_dim=config.output_dim,
        num_segments=config.num_segments,
        gnn_layers=config.gnn_layers,
        gnn_heads=config.gnn_heads,
        ra2r_layers=config.ra2r_layers,
        ra2r_heads=config.ra2r_heads,
        dropout=config.dropout
    )
    
    return model


if __name__ == '__main__':
    # Test the proposed model
    batch_size = 4
    num_segments = 32
    feature_dim = 2048
    
    # Create dummy input
    x = torch.randn(batch_size, num_segments, feature_dim)
    
    # Test full model
    model = ProposedModel()
    output = model(x, return_features=True)
    
    print("Proposed Model Test:")
    print(f"  Input shape: {x.shape}")
    print(f"  Scores shape: {output['scores'].shape}")
    print(f"  Features shape: {output['features'].shape}")
    print(f"  Adjacency shape: {output['adj'].shape}")
    print(f"  Score range: [{output['scores'].min().item():.4f}, {output['scores'].max().item():.4f}]")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Test lightweight model
    print("\nLightweight Model Test:")
    light_model = LightweightModel()
    output_light = light_model(x)
    print(f"  Scores shape: {output_light['scores'].shape}")
    
    light_params = sum(p.numel() for p in light_model.parameters())
    print(f"  Parameters: {light_params:,}")
