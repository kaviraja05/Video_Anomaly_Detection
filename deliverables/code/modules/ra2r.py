"""
Relation-Aware Reasoning (RA2R) module for Video Anomaly Detection.

Implements cross-segment relation reasoning to capture high-order
dependencies between video segments for anomaly detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class RelationEncoder(nn.Module):
    """
    Encodes pairwise relations between segments.
    
    Computes relation representations for each pair of segments,
    capturing how they relate to each other.
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        relation_dim: int = 64,
        dropout: float = 0.3
    ):
        """
        Initialize the relation encoder.
        
        Args:
            input_dim: Input feature dimension.
            relation_dim: Dimension of relation representations.
            dropout: Dropout rate.
        """
        super().__init__()
        
        self.relation_dim = relation_dim
        
        # Relation computation layers
        self.fc_concat = nn.Linear(input_dim * 2, relation_dim * 2)
        self.fc_diff = nn.Linear(input_dim, relation_dim)
        self.fc_prod = nn.Linear(input_dim, relation_dim)
        
        # Combine different relation types
        self.combine = nn.Sequential(
            nn.Linear(relation_dim * 4, relation_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(relation_dim * 2, relation_dim)
        )
        
        self.layer_norm = nn.LayerNorm(relation_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise relation representations.
        
        Args:
            x: Segment features of shape (B, N, D).
            
        Returns:
            Relations of shape (B, N, N, relation_dim).
        """
        B, N, D = x.shape
        
        # Expand for pairwise computation
        x_i = x.unsqueeze(2).expand(-1, -1, N, -1)  # (B, N, N, D)
        x_j = x.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, N, D)
        
        # Different relation types
        # 1. Concatenation-based
        concat = torch.cat([x_i, x_j], dim=-1)  # (B, N, N, 2D)
        r_concat = F.relu(self.fc_concat(concat))  # (B, N, N, 2*relation_dim)
        
        # 2. Difference-based
        diff = x_i - x_j  # (B, N, N, D)
        r_diff = F.relu(self.fc_diff(diff))  # (B, N, N, relation_dim)
        
        # 3. Element-wise product
        prod = x_i * x_j  # (B, N, N, D)
        r_prod = F.relu(self.fc_prod(prod))  # (B, N, N, relation_dim)
        
        # Combine all relation representations
        relations = torch.cat([r_concat, r_diff, r_prod], dim=-1)  # (B, N, N, 4*relation_dim)
        relations = self.combine(relations)  # (B, N, N, relation_dim)
        relations = self.layer_norm(relations)
        
        return relations


class RelationReasoning(nn.Module):
    """
    Multi-layer relation reasoning module.
    
    Propagates information through the relation graph to capture
    high-order dependencies between segments.
    """
    
    def __init__(
        self,
        node_dim: int = 128,
        relation_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.3
    ):
        """
        Initialize the relation reasoning module.
        
        Args:
            node_dim: Node feature dimension.
            relation_dim: Relation representation dimension.
            num_heads: Number of attention heads.
            dropout: Dropout rate.
        """
        super().__init__()
        
        self.node_dim = node_dim
        self.relation_dim = relation_dim
        self.num_heads = num_heads
        self.head_dim = node_dim // num_heads
        
        # Query, Key, Value for attention
        self.W_q = nn.Linear(node_dim, node_dim)
        self.W_k = nn.Linear(node_dim, node_dim)
        self.W_v = nn.Linear(node_dim, node_dim)
        
        # Relation-aware bias
        self.relation_bias = nn.Linear(relation_dim, num_heads)
        
        # Output projection
        self.W_out = nn.Linear(node_dim, node_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(node_dim)
        
        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(node_dim, node_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim * 4, node_dim),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(node_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        relations: torch.Tensor,
        adj: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Perform relation-aware reasoning.
        
        Args:
            x: Node features of shape (B, N, node_dim).
            relations: Relation features of shape (B, N, N, relation_dim).
            adj: Optional adjacency mask of shape (B, N, N).
            
        Returns:
            Updated node features of shape (B, N, node_dim).
        """
        B, N, D = x.shape
        
        # Compute Q, K, V
        Q = self.W_q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        # Q, K, V: (B, num_heads, N, head_dim)
        
        # Compute attention scores
        attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # attn: (B, num_heads, N, N)
        
        # Add relation-aware bias
        rel_bias = self.relation_bias(relations)  # (B, N, N, num_heads)
        rel_bias = rel_bias.permute(0, 3, 1, 2)  # (B, num_heads, N, N)
        attn = attn + rel_bias
        
        # Apply adjacency mask if provided
        if adj is not None:
            adj_mask = adj.unsqueeze(1)  # (B, 1, N, N)
            attn = attn.masked_fill(adj_mask < 0.5, -1e9)
        
        # Softmax and dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, V)  # (B, num_heads, N, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        
        # Output projection
        out = self.W_out(out)
        out = self.dropout(out)
        
        # Residual connection and layer norm
        x = self.layer_norm(out + x)
        
        # Feedforward network
        out = self.ffn(x)
        x = self.ffn_norm(out + x)
        
        return x


class RelationAwareReasoning(nn.Module):
    """
    Relation-Aware Reasoning (RA2R) module.
    
    Main module that combines relation encoding with multi-layer
    reasoning for cross-segment dependency modeling.
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        relation_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        """
        Initialize the RA2R module.
        
        Args:
            input_dim: Input feature dimension.
            relation_dim: Relation representation dimension.
            num_heads: Number of attention heads.
            num_layers: Number of reasoning layers.
            dropout: Dropout rate.
        """
        super().__init__()
        
        self.num_layers = num_layers
        
        # Relation encoder
        self.relation_encoder = RelationEncoder(
            input_dim=input_dim,
            relation_dim=relation_dim,
            dropout=dropout
        )
        
        # Reasoning layers
        self.reasoning_layers = nn.ModuleList([
            RelationReasoning(
                node_dim=input_dim,
                relation_dim=relation_dim,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Relation update (optional: update relations across layers)
        self.relation_update = nn.Sequential(
            nn.Linear(relation_dim + input_dim, relation_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        adj: Optional[torch.Tensor] = None,
        return_relations: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through RA2R.
        
        Args:
            x: Input features of shape (B, N, D).
            adj: Optional adjacency matrix of shape (B, N, N).
            return_relations: If True, also return final relations.
            
        Returns:
            x: Updated features of shape (B, N, D).
            relations: Final relation representations (optional).
        """
        B, N, D = x.shape
        
        # Encode initial relations
        relations = self.relation_encoder(x)  # (B, N, N, relation_dim)
        
        # Multi-layer reasoning
        for layer in self.reasoning_layers:
            x = layer(x, relations, adj)
            
            # Update relations based on new node features
            # Expand x for pairwise
            x_i = x.unsqueeze(2).expand(-1, -1, N, -1)  # (B, N, N, D)
            x_j = x.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, N, D)
            combined = (x_i + x_j) / 2  # (B, N, N, D)
            
            # Concatenate with current relations and update
            rel_input = torch.cat([relations, combined], dim=-1)
            relations = self.relation_update(rel_input)
        
        if return_relations:
            return x, relations
        return x, None


class HierarchicalRA2R(nn.Module):
    """
    Hierarchical RA2R for multi-scale reasoning.
    
    Applies RA2R at multiple temporal resolutions and fuses the results.
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        relation_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        scales: Tuple[int, ...] = (1, 2),
        dropout: float = 0.3
    ):
        """
        Initialize Hierarchical RA2R.
        
        Args:
            input_dim: Input feature dimension.
            relation_dim: Relation dimension.
            num_heads: Number of attention heads.
            num_layers: Number of reasoning layers per scale.
            scales: Temporal scales for hierarchical reasoning.
            dropout: Dropout rate.
        """
        super().__init__()
        
        self.scales = scales
        
        # RA2R module for each scale
        self.ra2r_modules = nn.ModuleList([
            RelationAwareReasoning(
                input_dim=input_dim,
                relation_dim=relation_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout
            ) for _ in scales
        ])
        
        # Feature fusion
        self.fuse = nn.Sequential(
            nn.Linear(input_dim * len(scales), input_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 2, input_dim)
        )
        self.layer_norm = nn.LayerNorm(input_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        adj: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, None]:
        """
        Forward pass through hierarchical RA2R.
        
        Args:
            x: Input features of shape (B, N, D).
            adj: Optional adjacency matrix.
            
        Returns:
            Fused features of shape (B, N, D).
        """
        B, N, D = x.shape
        outputs = []
        
        for scale, ra2r in zip(self.scales, self.ra2r_modules):
            if scale == 1:
                out, _ = ra2r(x, adj)
            else:
                # Pool to coarser resolution
                x_pooled = F.avg_pool1d(
                    x.transpose(1, 2),
                    kernel_size=scale,
                    stride=scale
                ).transpose(1, 2)
                
                # Process at coarser scale
                out_pooled, _ = ra2r(x_pooled)
                
                # Upsample back
                out = F.interpolate(
                    out_pooled.transpose(1, 2),
                    size=N,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)
            
            outputs.append(out)
        
        # Fuse multi-scale features
        fused = torch.cat(outputs, dim=-1)
        fused = self.fuse(fused)
        fused = self.layer_norm(fused + x)  # Residual
        
        return fused, None


if __name__ == '__main__':
    # Test RA2R modules
    batch_size = 4
    num_segments = 32
    feature_dim = 128
    relation_dim = 64
    
    x = torch.randn(batch_size, num_segments, feature_dim)
    
    # Test basic RA2R
    ra2r = RelationAwareReasoning(
        input_dim=feature_dim,
        relation_dim=relation_dim,
        num_layers=2
    )
    out, relations = ra2r(x, return_relations=True)
    print(f"RA2R output shape: {out.shape}")
    print(f"Relations shape: {relations.shape}")
    
    # Test Hierarchical RA2R
    hier_ra2r = HierarchicalRA2R(
        input_dim=feature_dim,
        relation_dim=relation_dim
    )
    out_hier, _ = hier_ra2r(x)
    print(f"Hierarchical RA2R output shape: {out_hier.shape}")
