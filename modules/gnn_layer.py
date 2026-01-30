"""
Graph Neural Network layer for Video Anomaly Detection.

Implements message passing on graphs constructed from video segments,
enabling relation-aware feature learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class GraphConvLayer(nn.Module):
    """
    Graph Convolutional Layer with attention-based message passing.
    
    Performs message passing between video segments based on their
    pairwise similarities (adjacency matrix).
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_heads: int = 4,
        dropout: float = 0.3,
        use_bias: bool = True
    ):
        """
        Initialize the graph convolution layer.
        
        Args:
            input_dim: Input feature dimension.
            output_dim: Output feature dimension.
            num_heads: Number of attention heads.
            dropout: Dropout rate.
            use_bias: Whether to use bias in linear layers.
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        
        assert output_dim % num_heads == 0, "output_dim must be divisible by num_heads"
        
        # Linear transformations for message passing
        self.W_query = nn.Linear(input_dim, output_dim, bias=use_bias)
        self.W_key = nn.Linear(input_dim, output_dim, bias=use_bias)
        self.W_value = nn.Linear(input_dim, output_dim, bias=use_bias)
        
        # Output projection
        self.W_out = nn.Linear(output_dim, output_dim, bias=use_bias)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)
        
        # Scaling factor for attention
        self.scale = math.sqrt(self.head_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        adj: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with message passing.
        
        Args:
            x: Node features of shape (B, N, D) where N = num_segments.
            adj: Adjacency matrix of shape (B, N, N) or None.
                 If None, computes attention-based adjacency.
            mask: Optional mask of shape (B, N) for padding.
            
        Returns:
            Updated node features of shape (B, N, output_dim).
        """
        B, N, D = x.shape
        
        # Compute query, key, value
        Q = self.W_query(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_key(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_value(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        # Q, K, V: (B, num_heads, N, head_dim)
        
        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, num_heads, N, N)
        
        # Incorporate adjacency matrix if provided
        if adj is not None:
            # Expand adj for multi-head: (B, 1, N, N)
            adj_expanded = adj.unsqueeze(1)
            # Use adjacency to mask or weight attention
            attn_scores = attn_scores * adj_expanded + (1 - adj_expanded) * (-1e9)
        
        # Apply mask if provided
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
            attn_scores = attn_scores.masked_fill(~mask_expanded, -1e9)
        
        # Compute attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Message aggregation
        out = torch.matmul(attn_weights, V)  # (B, num_heads, N, head_dim)
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(B, N, -1)  # (B, N, output_dim)
        out = self.W_out(out)
        out = self.dropout(out)
        
        # Residual connection and layer norm
        if D == self.output_dim:
            out = self.layer_norm(out + x)
        else:
            out = self.layer_norm(out)
        
        return out


class GNNBlock(nn.Module):
    """
    Multi-layer Graph Neural Network block.
    
    Stacks multiple GraphConvLayers with feedforward networks
    for deeper message passing.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3
    ):
        """
        Initialize the GNN block.
        
        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden dimension.
            output_dim: Output feature dimension.
            num_layers: Number of GNN layers.
            num_heads: Number of attention heads.
            dropout: Dropout rate.
        """
        super().__init__()
        
        self.layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        
        for i in range(num_layers):
            self.layers.append(GraphConvLayer(
                input_dim=dims[i],
                output_dim=dims[i + 1],
                num_heads=num_heads,
                dropout=dropout
            ))
            
            # Feedforward network after each GNN layer
            self.ffn_layers.append(nn.Sequential(
                nn.Linear(dims[i + 1], dims[i + 1] * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dims[i + 1] * 2, dims[i + 1]),
                nn.Dropout(dropout)
            ))
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(dims[i + 1]) for i in range(num_layers)
        ])
        
    def forward(
        self,
        x: torch.Tensor,
        adj: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through all GNN layers.
        
        Args:
            x: Node features of shape (B, N, D).
            adj: Adjacency matrix of shape (B, N, N) or None.
            
        Returns:
            Updated node features of shape (B, N, output_dim).
        """
        for gnn, ffn, norm in zip(self.layers, self.ffn_layers, self.layer_norms):
            # Graph convolution
            x = gnn(x, adj)
            
            # Feedforward with residual
            residual = x
            x = ffn(x)
            x = norm(x + residual)
        
        return x


def compute_adjacency(
    features: torch.Tensor,
    method: str = 'cosine',
    threshold: float = 0.5,
    k_neighbors: Optional[int] = None,
    add_self_loops: bool = True
) -> torch.Tensor:
    """
    Compute adjacency matrix from segment features.
    
    Args:
        features: Segment features of shape (B, N, D).
        method: Similarity method ('cosine', 'euclidean', 'dot').
        threshold: Threshold for binary adjacency (0-1).
        k_neighbors: If provided, use k-nearest neighbors.
        add_self_loops: Whether to add self-connections.
        
    Returns:
        Adjacency matrix of shape (B, N, N).
    """
    B, N, D = features.shape
    
    if method == 'cosine':
        # Normalize features
        features_norm = F.normalize(features, p=2, dim=-1)
        # Compute cosine similarity
        similarity = torch.bmm(features_norm, features_norm.transpose(1, 2))
        
    elif method == 'euclidean':
        # Compute pairwise distances
        diff = features.unsqueeze(2) - features.unsqueeze(1)  # (B, N, N, D)
        distances = torch.norm(diff, p=2, dim=-1)  # (B, N, N)
        # Convert to similarity (inverse distance)
        similarity = 1 / (1 + distances)
        
    elif method == 'dot':
        similarity = torch.bmm(features, features.transpose(1, 2))
        # Scale by dimension
        similarity = similarity / math.sqrt(D)
        similarity = torch.sigmoid(similarity)
        
    else:
        raise ValueError(f"Unknown similarity method: {method}")
    
    # Apply threshold or k-nearest neighbors
    if k_neighbors is not None:
        # Keep only top-k connections per node
        _, indices = similarity.topk(k_neighbors, dim=-1)
        adj = torch.zeros_like(similarity)
        adj.scatter_(-1, indices, 1.0)
        # Make symmetric
        adj = (adj + adj.transpose(1, 2)) / 2
        adj = (adj > 0).float()
    else:
        # Apply threshold
        adj = (similarity > threshold).float()
    
    # Add self-loops
    if add_self_loops:
        eye = torch.eye(N, device=features.device).unsqueeze(0).expand(B, -1, -1)
        adj = torch.clamp(adj + eye, max=1.0)
    
    return adj


def compute_temporal_adjacency(
    num_segments: int,
    window_size: int = 3,
    device: torch.device = None
) -> torch.Tensor:
    """
    Compute temporal adjacency based on segment proximity.
    
    Connects each segment to its temporal neighbors within a window.
    
    Args:
        num_segments: Number of segments (N).
        window_size: Size of temporal window (connects to ±window_size neighbors).
        device: Device to create tensor on.
        
    Returns:
        Temporal adjacency matrix of shape (N, N).
    """
    adj = torch.zeros(num_segments, num_segments, device=device)
    
    for i in range(num_segments):
        start = max(0, i - window_size)
        end = min(num_segments, i + window_size + 1)
        adj[i, start:end] = 1.0
    
    return adj


if __name__ == '__main__':
    # Test GNN layer
    batch_size = 4
    num_segments = 32
    feature_dim = 128
    
    # Create dummy features
    features = torch.randn(batch_size, num_segments, feature_dim)
    
    # Compute adjacency
    adj = compute_adjacency(features, method='cosine', threshold=0.3)
    print(f"Adjacency shape: {adj.shape}")
    print(f"Avg connections per node: {adj.sum(-1).mean().item():.2f}")
    
    # Test GNN block
    gnn = GNNBlock(
        input_dim=feature_dim,
        hidden_dim=feature_dim,
        output_dim=feature_dim,
        num_layers=2
    )
    
    output = gnn(features, adj)
    print(f"GNN output shape: {output.shape}")
