"""
Base model components for Video Anomaly Detection.

Provides feature embedding, temporal modeling, and anomaly scoring layers
that serve as building blocks for the full model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class FeatureEmbedding(nn.Module):
    """
    Feature embedding module to project I3D features to a lower dimension.
    
    Takes raw I3D features (2048-dim) and projects them through FC layers
    with non-linearities to produce compact representations.
    """
    
    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 512,
        output_dim: int = 128,
        dropout: float = 0.6
    ):
        """
        Initialize the feature embedding module.
        
        Args:
            input_dim: Input feature dimension (I3D = 2048).
            hidden_dim: Hidden layer dimension.
            output_dim: Output embedding dimension.
            dropout: Dropout rate.
        """
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features of shape (B, T, D) or (B*T, D).
            
        Returns:
            Embedded features of shape (B, T, output_dim) or (B*T, output_dim).
        """
        # Handle 3D input (batch, segments, features)
        original_shape = x.shape
        if len(original_shape) == 3:
            B, T, D = original_shape
            x = x.view(B * T, D)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.layer_norm(x)
        
        # Reshape back if needed
        if len(original_shape) == 3:
            x = x.view(B, T, -1)
        
        return x


class TemporalModule(nn.Module):
    """
    Temporal modeling module using 1D convolutions.
    
    Captures local temporal patterns in the video segments using
    multi-scale convolutional kernels.
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        kernel_sizes: Tuple[int, ...] = (3, 5),
        dropout: float = 0.3
    ):
        """
        Initialize the temporal module.
        
        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden/output dimension.
            num_layers: Number of convolutional layers per kernel size.
            kernel_sizes: Tuple of kernel sizes for multi-scale modeling.
            dropout: Dropout rate.
        """
        super().__init__()
        
        self.convs = nn.ModuleList()
        
        for kernel_size in kernel_sizes:
            layers = []
            in_channels = input_dim
            
            for i in range(num_layers):
                out_channels = hidden_dim
                padding = kernel_size // 2
                
                layers.extend([
                    nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout)
                ])
                in_channels = out_channels
            
            self.convs.append(nn.Sequential(*layers))
        
        # Combine multi-scale features
        self.combine = nn.Linear(hidden_dim * len(kernel_sizes), hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input of shape (B, T, D).
            
        Returns:
            Temporally modeled features of shape (B, T, hidden_dim).
        """
        # x: (B, T, D) -> (B, D, T) for conv1d
        x_conv = x.transpose(1, 2)
        
        # Apply multi-scale convolutions
        conv_outputs = []
        for conv in self.convs:
            out = conv(x_conv)  # (B, hidden_dim, T)
            conv_outputs.append(out.transpose(1, 2))  # (B, T, hidden_dim)
        
        # Concatenate and combine
        combined = torch.cat(conv_outputs, dim=-1)  # (B, T, hidden_dim * num_kernels)
        combined = self.combine(combined)  # (B, T, hidden_dim)
        combined = self.layer_norm(combined)
        
        # Residual connection
        return combined + x if x.shape[-1] == combined.shape[-1] else combined


class TemporalTransformer(nn.Module):
    """
    Temporal modeling using Transformer encoder.
    
    Uses self-attention to model long-range temporal dependencies
    between video segments.
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.3
    ):
        """
        Initialize the temporal transformer.
        
        Args:
            input_dim: Feature dimension.
            num_heads: Number of attention heads.
            num_layers: Number of transformer layers.
            dim_feedforward: Dimension of feedforward network.
            dropout: Dropout rate.
        """
        super().__init__()
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(input_dim, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input of shape (B, T, D).
            
        Returns:
            Transformed features of shape (B, T, D).
        """
        x = self.pos_encoding(x)
        return self.transformer(x)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class AnomalyScorer(nn.Module):
    """
    Anomaly scoring head.
    
    Takes segment features and produces per-segment anomaly scores
    in the range [0, 1].
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 64,
        dropout: float = 0.3
    ):
        """
        Initialize the anomaly scorer.
        
        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden layer dimension.
            dropout: Dropout rate.
        """
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features of shape (B, T, D).
            
        Returns:
            Anomaly scores of shape (B, T, 1) or squeezed (B, T).
        """
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        scores = torch.sigmoid(x)
        return scores.squeeze(-1)


class BaseModel(nn.Module):
    """
    Base model for video anomaly detection.
    
    Combines feature embedding, temporal modeling, and anomaly scoring
    without graph-based components.
    """
    
    def __init__(
        self,
        feature_dim: int = 2048,
        hidden_dim: int = 512,
        output_dim: int = 128,
        dropout: float = 0.6,
        use_transformer: bool = False
    ):
        """
        Initialize the base model.
        
        Args:
            feature_dim: I3D feature dimension.
            hidden_dim: Hidden layer dimension.
            output_dim: Embedding dimension.
            dropout: Dropout rate.
            use_transformer: Use Transformer instead of Conv1D.
        """
        super().__init__()
        
        # Feature embedding
        self.embedding = FeatureEmbedding(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout
        )
        
        # Temporal modeling
        if use_transformer:
            self.temporal = TemporalTransformer(
                input_dim=output_dim,
                num_heads=4,
                num_layers=2,
                dropout=dropout / 2
            )
        else:
            self.temporal = TemporalModule(
                input_dim=output_dim,
                hidden_dim=output_dim,
                dropout=dropout / 2
            )
        
        # Anomaly scoring
        self.scorer = AnomalyScorer(
            input_dim=output_dim,
            hidden_dim=output_dim // 2,
            dropout=dropout / 2
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: I3D features of shape (B, T, feature_dim).
            
        Returns:
            Anomaly scores of shape (B, T).
        """
        # Embed features
        x = self.embedding(x)  # (B, T, output_dim)
        
        # Temporal modeling
        x = self.temporal(x)  # (B, T, output_dim)
        
        # Score anomalies
        scores = self.scorer(x)  # (B, T)
        
        return scores
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get intermediate features (useful for GNN integration).
        
        Args:
            x: I3D features of shape (B, T, feature_dim).
            
        Returns:
            Embedded features of shape (B, T, output_dim).
        """
        x = self.embedding(x)
        x = self.temporal(x)
        return x


if __name__ == '__main__':
    # Test the base model
    model = BaseModel()
    
    # Dummy input: batch of 4 videos, 32 segments each, 2048 features
    x = torch.randn(4, 32, 2048)
    
    scores = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {scores.shape}")
    print(f"Score range: [{scores.min().item():.4f}, {scores.max().item():.4f}]")
