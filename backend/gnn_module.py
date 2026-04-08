import torch
import torch.nn.functional as F
import math
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from modules.gnn_layer import GNNBlock, compute_adjacency

class GraphNeuralNetworkEnhancer:
    """
    Lightweight temporal graph reasoning module for Video Anomaly Detection.
    Captures temporal relationships between video segments to refine anomaly scores.
    """
    
    def __init__(self, feature_dim=128, gnn_layers=2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gnn = GNNBlock(
            input_dim=feature_dim,
            hidden_dim=feature_dim,
            output_dim=feature_dim,
            num_layers=gnn_layers
        ).to(self.device)
        self.gnn.eval() # Mostly used in inference

    def refine_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Process features through the GNN to capture temporal relationships.
        Args:
            features: Tensor of shape (B, T, D)
        Returns:
            Refined features of shape (B, T, D)
        """
        features = features.to(self.device)
        adj = compute_adjacency(features, method='cosine', threshold=0.5)
        
        with torch.no_grad():
            refined = self.gnn(features, adj)
        return refined
