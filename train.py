"""
Training script for Video Anomaly Detection.

Implements the complete training pipeline with:
- Configuration loading
- Dataset preparation
- Model initialization
- Training loop with MIL loss
- Checkpoint saving
- Logging
"""

import os
import sys
import argparse
import random
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from utils.config import get_config, update_config
from utils.dataloader import get_dataloaders
from models.proposed_model import ProposedModel, build_model
from modules.mil_loss import MILLoss, get_loss_fn


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Video Anomaly Detection Model')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=None,
                        help='Weight decay for optimizer')
    
    # Model parameters
    parser.add_argument('--hidden_dim', type=int, default=None,
                        help='Hidden dimension')
    parser.add_argument('--output_dim', type=int, default=None,
                        help='Output embedding dimension')
    parser.add_argument('--dropout', type=float, default=None,
                        help='Dropout rate')
    
    # Loss parameters
    parser.add_argument('--mil_topk', type=int, default=None,
                        help='Top-k for MIL loss')
    parser.add_argument('--loss_type', type=str, default='mil',
                        choices=['mil', 'contrastive', 'focal'],
                        help='Type of loss function')
    
    # Other
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    train_loader,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    config
) -> dict:
    """
    Train for one epoch.
    
    Args:
        model: The model to train.
        train_loader: Training data loader.
        loss_fn: Loss function.
        optimizer: Optimizer.
        device: Device to train on.
        epoch: Current epoch number.
        config: Configuration object.
        
    Returns:
        Dictionary with training metrics.
    """
    model.train()
    
    total_loss = 0.0
    loss_components = {'ranking': 0.0, 'normal': 0.0, 'smoothness': 0.0}
    num_batches = 0
    
    for batch_idx, batch in enumerate(train_loader):
        features = batch['features'].to(device)  # (B, T, D)
        labels = batch['label'].to(device)  # (B,)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(features, return_features=True)
        scores = output['scores']  # (B, T)
        
        # Compute loss
        if config.loss_type == 'contrastive' and 'features' in output:
            # For contrastive loss that uses features
            loss, loss_dict = loss_fn(scores, labels, output.get('features'))
        else:
            loss, loss_dict = loss_fn(scores, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Accumulate metrics
        total_loss += loss.item()
        for key in loss_components:
            if key in loss_dict:
                loss_components[key] += loss_dict[key]
        num_batches += 1
        
        # Logging
        if (batch_idx + 1) % config.log_interval == 0:
            avg_loss = total_loss / num_batches
            print(f'  Epoch [{epoch}] Batch [{batch_idx + 1}/{len(train_loader)}] '
                  f'Loss: {avg_loss:.4f}')
    
    # Compute averages
    metrics = {
        'loss': total_loss / num_batches,
    }
    for key, value in loss_components.items():
        metrics[f'loss_{key}'] = value / num_batches
    
    return metrics


def validate(
    model: nn.Module,
    val_loader,
    loss_fn: nn.Module,
    device: torch.device
) -> dict:
    """
    Validate the model.
    
    Args:
        model: The model to validate.
        val_loader: Validation data loader.
        loss_fn: Loss function.
        device: Device.
        
    Returns:
        Dictionary with validation metrics.
    """
    model.eval()
    
    total_loss = 0.0
    all_scores = []
    all_labels = []
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            features = batch['features'].to(device)
            labels = batch['label'].to(device)
            
            output = model(features)
            scores = output['scores']
            
            loss, _ = loss_fn(scores, labels)
            
            total_loss += loss.item()
            all_scores.extend(scores.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
            num_batches += 1
    
    metrics = {
        'loss': total_loss / max(num_batches, 1),
    }
    
    return metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: dict,
    config,
    is_best: bool = False
):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': config.to_dict()
    }
    
    # Save latest
    latest_path = os.path.join(config.checkpoint_dir, 'latest_model.pth')
    torch.save(checkpoint, latest_path)
    
    # Save periodic
    if (epoch + 1) % config.save_interval == 0:
        epoch_path = os.path.join(config.checkpoint_dir, f'model_epoch_{epoch + 1}.pth')
        torch.save(checkpoint, epoch_path)
    
    # Save best
    if is_best:
        best_path = os.path.join(config.checkpoint_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
        print(f'  Saved best model with loss: {metrics["loss"]:.4f}')


def load_checkpoint(path: str, model: nn.Module, optimizer: optim.Optimizer = None):
    """Load model checkpoint."""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint.get('epoch', 0), checkpoint.get('metrics', {})


def train(config):
    """
    Main training function.
    
    Args:
        config: Configuration object.
    """
    print("=" * 60)
    print("Video Anomaly Detection Training")
    print("=" * 60)
    
    # Set device
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seed
    set_seed(config.seed)
    
    # Create data loaders
    print("\nLoading data...")
    train_loader, test_loader = get_dataloaders(config)
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Test samples: {len(test_loader)}")
    
    # Create model
    print("\nBuilding model...")
    model = build_model(config)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Create loss function
    loss_fn = get_loss_fn(
        loss_type=config.loss_type,
        topk=config.mil_topk,
        margin=config.mil_margin,
        smoothness_weight=config.smoothness_weight,
        sparsity_weight=config.sparsity_weight
    )
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=config.learning_rate / 100
    )
    
    # TensorBoard writer
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(config.log_dir, f'run_{timestamp}')
    writer = SummaryWriter(log_path)
    print(f"\nLogs: {log_path}")
    
    # Training loop
    print("\nStarting training...")
    best_loss = float('inf')
    start_epoch = 0
    
    for epoch in range(start_epoch, config.epochs):
        print(f"\nEpoch {epoch + 1}/{config.epochs}")
        print("-" * 40)
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device, epoch + 1, config
        )
        
        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Log to TensorBoard
        writer.add_scalar('train/loss', train_metrics['loss'], epoch)
        writer.add_scalar('train/lr', current_lr, epoch)
        for key, value in train_metrics.items():
            if key.startswith('loss_'):
                writer.add_scalar(f'train/{key}', value, epoch)
        
        # Print metrics
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Save checkpoint
        is_best = train_metrics['loss'] < best_loss
        if is_best:
            best_loss = train_metrics['loss']
        
        save_checkpoint(model, optimizer, epoch, train_metrics, config, is_best)
    
    # Close writer
    writer.close()
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best training loss: {best_loss:.4f}")
    print(f"Checkpoints saved in: {config.checkpoint_dir}")
    print("=" * 60)


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Get base config
    config = get_config()
    
    # Update config with command line arguments
    updates = {}
    if args.epochs is not None:
        updates['epochs'] = args.epochs
    if args.batch_size is not None:
        updates['batch_size'] = args.batch_size
    if args.lr is not None:
        updates['learning_rate'] = args.lr
    if args.weight_decay is not None:
        updates['weight_decay'] = args.weight_decay
    if args.hidden_dim is not None:
        updates['hidden_dim'] = args.hidden_dim
    if args.output_dim is not None:
        updates['output_dim'] = args.output_dim
    if args.dropout is not None:
        updates['dropout'] = args.dropout
    if args.loss_type is not None:
        updates['loss_type'] = args.loss_type
    if args.mil_topk is not None:
        updates['mil_topk'] = args.mil_topk
    if args.device is not None:
        updates['device'] = args.device
    if args.seed is not None:
        updates['seed'] = args.seed
    
    if updates:
        update_config(**updates)
    
    # Run training
    train(config)


if __name__ == '__main__':
    main()
