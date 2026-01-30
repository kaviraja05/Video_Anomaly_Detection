"""
Multiple Instance Learning (MIL) Loss for Video Anomaly Detection.

Implements MIL-based loss functions for weakly supervised anomaly detection
where only video-level labels (normal/abnormal) are available.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MILLoss(nn.Module):
    """
    Multiple Instance Learning Loss for weakly supervised anomaly detection.
    
    Treats each video as a "bag" of segment instances:
    - Normal videos: all segments should have low anomaly scores
    - Abnormal videos: at least some segments should have high scores
    
    Uses top-k instance selection and margin ranking loss.
    """
    
    def __init__(
        self,
        topk: int = 3,
        margin: float = 1.0,
        smoothness_weight: float = 8e-5,
        sparsity_weight: float = 8e-5
    ):
        """
        Initialize the MIL loss.
        
        Args:
            topk: Number of top instances to select per bag.
            margin: Margin for ranking loss between normal and abnormal.
            smoothness_weight: Weight for temporal smoothness regularization.
            sparsity_weight: Weight for sparsity regularization.
        """
        super().__init__()
        
        self.topk = topk
        self.margin = margin
        self.smoothness_weight = smoothness_weight
        self.sparsity_weight = sparsity_weight
        
        # Margin ranking loss
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        
    def forward(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute MIL loss.
        
        Args:
            scores: Anomaly scores of shape (B, T) where B is batch size
                    and T is number of segments.
            labels: Video-level labels of shape (B,), 0=normal, 1=abnormal.
            
        Returns:
            loss: Total loss value.
            loss_dict: Dictionary with individual loss components.
        """
        B, T = scores.shape
        device = scores.device
        
        # Separate normal and abnormal samples
        normal_mask = labels == 0
        abnormal_mask = labels == 1
        
        normal_scores = scores[normal_mask]  # (N_normal, T)
        abnormal_scores = scores[abnormal_mask]  # (N_abnormal, T)
        
        # Initialize losses
        ranking_loss = torch.tensor(0.0, device=device)
        normal_loss = torch.tensor(0.0, device=device)
        smoothness_loss = torch.tensor(0.0, device=device)
        sparsity_loss = torch.tensor(0.0, device=device)
        
        # =====================================================================
        # 1. Ranking Loss: max(abnormal) > max(normal) + margin
        # =====================================================================
        if normal_scores.shape[0] > 0 and abnormal_scores.shape[0] > 0:
            # Get top-k scores from each video
            normal_topk, _ = normal_scores.topk(self.topk, dim=1)  # (N_normal, k)
            abnormal_topk, _ = abnormal_scores.topk(self.topk, dim=1)  # (N_abnormal, k)
            
            # Average top-k
            normal_max = normal_topk.mean(dim=1)  # (N_normal,)
            abnormal_max = abnormal_topk.mean(dim=1)  # (N_abnormal,)
            
            # Pairwise ranking loss
            n_normal = normal_max.shape[0]
            n_abnormal = abnormal_max.shape[0]
            
            # Expand for pairwise comparison
            normal_expand = normal_max.unsqueeze(1).expand(-1, n_abnormal)  # (n_normal, n_abnormal)
            abnormal_expand = abnormal_max.unsqueeze(0).expand(n_normal, -1)  # (n_normal, n_abnormal)
            
            # Flatten for ranking loss
            normal_flat = normal_expand.flatten()
            abnormal_flat = abnormal_expand.flatten()
            target = torch.ones_like(normal_flat)  # abnormal > normal
            
            ranking_loss = self.ranking_loss(abnormal_flat, normal_flat, target)
        
        # =====================================================================
        # 2. Normal Video Constraint: all segments should be low
        # =====================================================================
        if normal_scores.shape[0] > 0:
            # Penalize high scores in normal videos
            normal_loss = normal_scores.mean()
        
        # =====================================================================
        # 3. Temporal Smoothness: adjacent segments should have similar scores
        # =====================================================================
        if self.smoothness_weight > 0:
            # Compute temporal difference
            diff = scores[:, 1:] - scores[:, :-1]  # (B, T-1)
            smoothness_loss = (diff ** 2).mean()
        
        # =====================================================================
        # 4. Sparsity: anomalies should be sparse (not constant high scores)
        # =====================================================================
        if self.sparsity_weight > 0 and abnormal_scores.shape[0] > 0:
            # Encourage sparsity through L1 regularization
            sparsity_loss = abnormal_scores.mean()
        
        # =====================================================================
        # Total Loss
        # =====================================================================
        total_loss = (
            ranking_loss + 
            normal_loss +
            self.smoothness_weight * smoothness_loss +
            self.sparsity_weight * sparsity_loss
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'ranking': ranking_loss.item() if isinstance(ranking_loss, torch.Tensor) else ranking_loss,
            'normal': normal_loss.item() if isinstance(normal_loss, torch.Tensor) else normal_loss,
            'smoothness': smoothness_loss.item() if isinstance(smoothness_loss, torch.Tensor) else smoothness_loss,
            'sparsity': sparsity_loss.item() if isinstance(sparsity_loss, torch.Tensor) else sparsity_loss
        }
        
        return total_loss, loss_dict


class ContrastiveMILLoss(nn.Module):
    """
    Contrastive MIL Loss with instance-level contrastive learning.
    
    Extends MIL loss with contrastive learning between normal and 
    abnormal segment representations.
    """
    
    def __init__(
        self,
        topk: int = 3,
        margin: float = 1.0,
        temperature: float = 0.1,
        contrastive_weight: float = 0.1,
        smoothness_weight: float = 8e-5,
        sparsity_weight: float = 8e-5
    ):
        """
        Initialize the Contrastive MIL loss.
        
        Args:
            topk: Number of top instances to select.
            margin: Margin for ranking loss.
            temperature: Temperature for contrastive loss.
            contrastive_weight: Weight for contrastive loss.
            smoothness_weight: Weight for smoothness regularization.
            sparsity_weight: Weight for sparsity regularization.
        """
        super().__init__()
        
        self.mil_loss = MILLoss(
            topk=topk,
            margin=margin,
            smoothness_weight=smoothness_weight,
            sparsity_weight=sparsity_weight
        )
        
        self.temperature = temperature
        self.contrastive_weight = contrastive_weight
        
    def forward(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute Contrastive MIL loss.
        
        Args:
            scores: Anomaly scores of shape (B, T).
            labels: Video-level labels of shape (B,).
            features: Segment features of shape (B, T, D) for contrastive loss.
            
        Returns:
            loss: Total loss value.
            loss_dict: Dictionary with individual loss components.
        """
        # Compute base MIL loss
        mil_loss, loss_dict = self.mil_loss(scores, labels)
        
        # Compute contrastive loss if features provided
        contrastive_loss = torch.tensor(0.0, device=scores.device)
        
        if features is not None and self.contrastive_weight > 0:
            contrastive_loss = self._compute_contrastive_loss(
                features, scores, labels
            )
            loss_dict['contrastive'] = contrastive_loss.item()
        
        total_loss = mil_loss + self.contrastive_weight * contrastive_loss
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict
    
    def _compute_contrastive_loss(
        self,
        features: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute instance-level contrastive loss.
        
        Pulls together normal instances and pushes apart normal from abnormal.
        """
        B, T, D = features.shape
        device = features.device
        
        # Get high-scoring segments from abnormal videos
        abnormal_mask = labels == 1
        normal_mask = labels == 0
        
        if abnormal_mask.sum() == 0 or normal_mask.sum() == 0:
            return torch.tensor(0.0, device=device)
        
        # Get features of suspicious segments (high scores in abnormal videos)
        abnormal_scores = scores[abnormal_mask]  # (N_abn, T)
        abnormal_features = features[abnormal_mask]  # (N_abn, T, D)
        
        # Top-k suspicious segments
        _, topk_idx = abnormal_scores.topk(self.mil_loss.topk, dim=1)  # (N_abn, k)
        
        # Gather top-k features
        topk_idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, D)  # (N_abn, k, D)
        suspicious_features = torch.gather(abnormal_features, 1, topk_idx_expanded)  # (N_abn, k, D)
        suspicious_features = suspicious_features.view(-1, D)  # (N_abn * k, D)
        
        # Get all normal segment features
        normal_features = features[normal_mask].view(-1, D)  # (N_norm * T, D)
        
        # Normalize features
        suspicious_norm = F.normalize(suspicious_features, p=2, dim=-1)
        normal_norm = F.normalize(normal_features, p=2, dim=-1)
        
        # Compute similarity
        sim = torch.mm(suspicious_norm, normal_norm.t()) / self.temperature  # (N_susp, N_norm)
        
        # Contrastive loss: push suspicious away from normal
        labels_cont = torch.zeros(sim.shape[0], dtype=torch.long, device=device)
        contrastive_loss = F.cross_entropy(
            -sim,  # Negative similarity = push apart
            labels_cont,
            reduction='mean'
        )
        
        return contrastive_loss


class FocalMILLoss(nn.Module):
    """
    Focal MIL Loss for handling class imbalance.
    
    Applies focal loss weighting to emphasize hard examples.
    """
    
    def __init__(
        self,
        topk: int = 3,
        margin: float = 1.0,
        gamma: float = 2.0,
        alpha: float = 0.25,
        smoothness_weight: float = 8e-5
    ):
        """
        Initialize the Focal MIL loss.
        
        Args:
            topk: Number of top instances.
            margin: Ranking margin.
            gamma: Focal loss gamma (focusing parameter).
            alpha: Class weight for positive class.
            smoothness_weight: Smoothness regularization weight.
        """
        super().__init__()
        
        self.topk = topk
        self.margin = margin
        self.gamma = gamma
        self.alpha = alpha
        self.smoothness_weight = smoothness_weight
        
    def forward(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute Focal MIL loss.
        
        Args:
            scores: Anomaly scores of shape (B, T).
            labels: Video-level labels of shape (B,).
            
        Returns:
            loss: Total loss value.
            loss_dict: Dictionary with loss components.
        """
        B, T = scores.shape
        device = scores.device
        
        normal_mask = labels == 0
        abnormal_mask = labels == 1
        
        losses = []
        
        # Process each video
        for i in range(B):
            video_scores = scores[i]  # (T,)
            is_abnormal = labels[i].item() == 1
            
            if is_abnormal:
                # For abnormal: high scores are correct, focus on low-scoring hard examples
                topk_scores, _ = video_scores.topk(self.topk)
                p = topk_scores.mean()  # Predicted anomaly probability
                
                # Focal weight: (1 - p)^gamma for abnormal
                focal_weight = ((1 - p) ** self.gamma)
                loss = -self.alpha * focal_weight * torch.log(p + 1e-8)
            else:
                # For normal: low scores are correct, focus on high-scoring hard examples
                topk_scores, _ = video_scores.topk(self.topk)
                p = topk_scores.mean()
                
                # Focal weight: p^gamma for normal
                focal_weight = (p ** self.gamma)
                loss = -(1 - self.alpha) * focal_weight * torch.log(1 - p + 1e-8)
            
            losses.append(loss)
        
        focal_loss = torch.stack(losses).mean()
        
        # Smoothness regularization
        diff = scores[:, 1:] - scores[:, :-1]
        smoothness_loss = (diff ** 2).mean()
        
        total_loss = focal_loss + self.smoothness_weight * smoothness_loss
        
        loss_dict = {
            'total': total_loss.item(),
            'focal': focal_loss.item(),
            'smoothness': smoothness_loss.item()
        }
        
        return total_loss, loss_dict


def get_loss_fn(loss_type: str = 'mil', **kwargs) -> nn.Module:
    """
    Factory function to get loss function by name.
    
    Args:
        loss_type: Type of loss ('mil', 'contrastive', 'focal').
        **kwargs: Additional arguments for the loss function.
        
    Returns:
        Loss function module.
    """
    loss_map = {
        'mil': MILLoss,
        'contrastive': ContrastiveMILLoss,
        'focal': FocalMILLoss
    }
    
    if loss_type not in loss_map:
        raise ValueError(f"Unknown loss type: {loss_type}. Available: {list(loss_map.keys())}")
    
    return loss_map[loss_type](**kwargs)


if __name__ == '__main__':
    # Test MIL loss
    batch_size = 10  # 5 normal + 5 abnormal
    num_segments = 32
    
    # Random scores
    scores = torch.rand(batch_size, num_segments)
    # First half normal, second half abnormal
    labels = torch.tensor([0] * 5 + [1] * 5)
    
    # Test MIL loss
    mil_loss_fn = MILLoss()
    loss, loss_dict = mil_loss_fn(scores, labels)
    print(f"MIL Loss: {loss.item():.4f}")
    print(f"Loss components: {loss_dict}")
    
    # Test with features for contrastive loss
    features = torch.randn(batch_size, num_segments, 128)
    contrastive_loss_fn = ContrastiveMILLoss()
    loss_c, loss_dict_c = contrastive_loss_fn(scores, labels, features)
    print(f"\nContrastive MIL Loss: {loss_c.item():.4f}")
    print(f"Loss components: {loss_dict_c}")
