"""
Dynamic Similarity Module (DSM) for Video Anomaly Detection.

Learns dynamic relationships between video segments through
attention-based similarity computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class DynamicSimilarityModule(nn.Module):
    """
    Dynamic Similarity Module (DSM).
    
    Computes dynamic, learnable similarity matrices between video segments
    that adapt based on content. This refined similarity is used as the
    adjacency matrix for GNN message passing.
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        num_heads: int = 4,
        temperature: float = 0.1,
        dropout: float = 0.3
    ):
        """
        Initialize the DSM.
        
        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden dimension for similarity computation.
            num_heads: Number of attention heads for multi-view similarity.
            temperature: Temperature for softmax (lower = sharper).
            dropout: Dropout rate.
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.temperature = temperature
        self.head_dim = hidden_dim // num_heads
        
        # Query and Key projections for similarity computation
        self.W_q = nn.Linear(input_dim, hidden_dim)
        self.W_k = nn.Linear(input_dim, hidden_dim)
        
        # Context-aware gating
        self.context_gate = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Similarity refinement
        self.refine = nn.Sequential(
            nn.Linear(num_heads, num_heads * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_heads * 2, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Learnable threshold
        self.threshold = nn.Parameter(torch.tensor(0.5))
        
    def forward(
        self,
        x: torch.Tensor,
        return_raw: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute dynamic similarity matrix.
        
        Args:
            x: Segment features of shape (B, N, D).
            return_raw: If True, also return raw similarity before thresholding.
            
        Returns:
            adj: Dynamic adjacency matrix of shape (B, N, N).
            raw_sim: Raw similarity scores (optional).
        """
        B, N, D = x.shape
        
        # Compute multi-head queries and keys
        Q = self.W_q(x).view(B, N, self.num_heads, self.head_dim)  # (B, N, H, head_dim)
        K = self.W_k(x).view(B, N, self.num_heads, self.head_dim)  # (B, N, H, head_dim)
        
        # Compute per-head similarities
        # Q: (B, N, H, d) -> (B, H, N, d)
        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        
        # Attention scores: (B, H, N, N)
        sim = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        sim = sim / self.temperature
        
        # Softmax over keys for each query
        sim_weights = F.softmax(sim, dim=-1)  # (B, H, N, N)
        
        # Combine multi-head similarities
        # Reshape to (B, N, N, H) for refinement
        sim_combined = sim_weights.permute(0, 2, 3, 1)
        
        # Refine to single adjacency value
        adj_raw = self.refine(sim_combined).squeeze(-1)  # (B, N, N)
        
        # Apply context-aware gating
        # For each pair (i, j), compute gate based on concatenated features
        x_i = x.unsqueeze(2).expand(-1, -1, N, -1)  # (B, N, N, D)
        x_j = x.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, N, D)
        pair_features = torch.cat([x_i, x_j], dim=-1)  # (B, N, N, 2D)
        
        gate = self.context_gate(pair_features).squeeze(-1)  # (B, N, N)
        
        # Gated adjacency
        adj_gated = adj_raw * gate
        
        # Apply learnable threshold
        adj = torch.sigmoid((adj_gated - self.threshold) * 10)
        
        # Make symmetric
        adj = (adj + adj.transpose(1, 2)) / 2
        
        # Add self-loops
        eye = torch.eye(N, device=x.device).unsqueeze(0).expand(B, -1, -1)
        adj = torch.clamp(adj + eye, max=1.0)
        
        if return_raw:
            return adj, adj_raw
        return adj, None


class TemporalDSM(nn.Module):
    """
    Temporal-aware Dynamic Similarity Module.
    
    Extends DSM to incorporate temporal position information,
    biasing connections toward temporally close segments.
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        max_segments: int = 32,
        num_heads: int = 4,
        temperature: float = 0.1,
        temporal_weight: float = 0.3,
        dropout: float = 0.3
    ):
        """
        Initialize the Temporal DSM.
        
        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden dimension.
            max_segments: Maximum number of segments (for positional encoding).
            num_heads: Number of attention heads.
            temperature: Softmax temperature.
            temporal_weight: Weight for temporal bias (0-1).
            dropout: Dropout rate.
        """
        super().__init__()
        
        self.dsm = DynamicSimilarityModule(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            temperature=temperature,
            dropout=dropout
        )
        
        self.temporal_weight = temporal_weight
        
        # Learnable temporal distance embedding
        self.temporal_embed = nn.Embedding(max_segments * 2, hidden_dim // 4)
        self.temporal_proj = nn.Linear(hidden_dim // 4, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute temporally-aware dynamic similarity.
        
        Args:
            x: Segment features of shape (B, N, D).
            
        Returns:
            adj: Dynamic adjacency with temporal bias.
            raw_sim: Raw similarity scores.
        """
        B, N, D = x.shape
        device = x.device
        
        # Compute content-based similarity
        adj_content, raw_sim = self.dsm(x, return_raw=True)
        
        # Compute temporal bias
        positions = torch.arange(N, device=device)
        # Distance matrix: (N, N)
        distances = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()
        # Offset to make all indices positive
        distances_offset = distances + N - 1  # Range: [0, 2N-2]
        
        # Embed temporal distances
        temporal_emb = self.temporal_embed(distances_offset)  # (N, N, hidden//4)
        temporal_bias = torch.sigmoid(self.temporal_proj(temporal_emb)).squeeze(-1)  # (N, N)
        
        # Expand for batch
        temporal_bias = temporal_bias.unsqueeze(0).expand(B, -1, -1)
        
        # Combine content and temporal similarities
        adj = (1 - self.temporal_weight) * adj_content + self.temporal_weight * temporal_bias
        
        return adj, raw_sim


class MultiScaleDSM(nn.Module):
    """
    Multi-scale Dynamic Similarity Module.
    
    Computes similarities at multiple temporal scales and combines them
    for richer relational modeling.
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        scales: Tuple[int, ...] = (1, 2, 4),
        num_heads: int = 4,
        dropout: float = 0.3
    ):
        """
        Initialize Multi-scale DSM.
        
        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden dimension.
            scales: Tuple of temporal scales for pooling.
            num_heads: Number of attention heads.
            dropout: Dropout rate.
        """
        super().__init__()
        
        self.scales = scales
        
        # DSM for each scale
        self.dsm_modules = nn.ModuleList([
            DynamicSimilarityModule(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in scales
        ])
        
        # Scale combination
        self.scale_weights = nn.Parameter(torch.ones(len(scales)) / len(scales))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """
        Compute multi-scale dynamic similarity.
        
        Args:
            x: Segment features of shape (B, N, D).
            
        Returns:
            adj: Combined multi-scale adjacency matrix.
        """
        B, N, D = x.shape
        adjacencies = []
        
        for scale, dsm in zip(self.scales, self.dsm_modules):
            if scale == 1:
                adj_scale, _ = dsm(x)
            else:
                # Pool features at this scale
                pooled = F.avg_pool1d(
                    x.transpose(1, 2),
                    kernel_size=scale,
                    stride=scale // 2 or 1
                ).transpose(1, 2)
                adj_scale, _ = dsm(pooled)
                # Upsample adjacency back to original resolution
                adj_scale = F.interpolate(
                    adj_scale.unsqueeze(1),
                    size=(N, N),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)
            
            adjacencies.append(adj_scale)
        
        # Weighted combination
        weights = F.softmax(self.scale_weights, dim=0)
        adj = sum(w * a for w, a in zip(weights, adjacencies))
        
        return adj, None


if __name__ == '__main__':
    # Test DSM modules
    batch_size = 4
    num_segments = 32
    feature_dim = 128
    
    x = torch.randn(batch_size, num_segments, feature_dim)
    
    # Test basic DSM
    dsm = DynamicSimilarityModule(input_dim=feature_dim)
    adj, raw = dsm(x, return_raw=True)
    print(f"DSM adjacency shape: {adj.shape}")
    print(f"DSM adjacency range: [{adj.min().item():.4f}, {adj.max().item():.4f}]")
    
    # Test Temporal DSM
    tdsm = TemporalDSM(input_dim=feature_dim)
    adj_t, _ = tdsm(x)
    print(f"Temporal DSM adjacency shape: {adj_t.shape}")
    
    # Test Multi-scale DSM
    msdsm = MultiScaleDSM(input_dim=feature_dim)
    adj_ms, _ = msdsm(x)
    print(f"Multi-scale DSM adjacency shape: {adj_ms.shape}")
