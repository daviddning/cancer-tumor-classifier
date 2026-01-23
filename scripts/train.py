"""
Medical Image Segmentation Training Pipeline

This module implements a complete training pipeline for medical image segmentation
using an Attention U-Net architecture with custom loss functions optimized for
small tumor detection.

Key Features:
- Custom loss functions for handling class imbalance and small tumors
- Mixed precision training for efficient GPU utilization
- Comprehensive metrics tracking and visualization
- Early stopping and learning rate scheduling
- Checkpoint management

Author: [DAVID NING]
Date: [01-22-2026]
"""



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from pathlib import Path
import time
import json
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import config
from data_processing import *
from model import AttentionUNet


class SmallTumorDiceLoss(nn.Module):
    """
    Custom Dice Loss with enhanced penalty for missed small tumors.
    
    The standard Dice loss is modified to give extra weight to small tumor regions
    that are incorrectly predicted, helping the model learn to detect smaller lesions.
    
    Attributes:
        smooth (float): Smoothing factor to avoid division by zero (default: 1.0)
        small_weight (float): Additional weight for small tumor regions (default: 2.0)
    """
    def __init__(self, smooth=1.0, small_weight=2.0):
        """
        Initialize the SmallTumorDiceLoss.
        
        Args:
            smooth (float): Smoothing constant to prevent division by zero
            small_weight (float): Weight multiplier for small tumor regions
        """
        super().__init__()
        self.smooth = smooth
        self.small_weight = small_weight
    
    def forward(self, pred, target):
        """
        Compute the small tumor-aware Dice loss.
        
        Args:
            pred (torch.Tensor): Predicted probabilities [B, 1, H, W], values in [0, 1]
            target (torch.Tensor): Ground truth binary masks [B, 1, H, W], values in {0, 1}
        
        Returns:
            torch.Tensor: Scalar loss value (1 - dice_score)
        """
        # Flatten tensors for easier computation
        pred = pred.view(-1)
        target = target.view(-1)

        # Calculate standard Dice score
        intersection = (pred * target).sum()
        dice_score = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        small_mask = (target > 0) & (pred < 0.5)
        
        # Calculate Dice score specifically for small tumor regions
        if small_mask.sum() > 0:
            small_intersection = (pred[small_mask] * target[small_mask]).sum()
            small_dice = (2. * small_intersection + self.smooth) / \
                        (pred[small_mask].sum() + target[small_mask].sum() + self.smooth)
            
            # Weighted combination of standard and small tumor Dice
            dice_score = (dice_score + self.small_weight * small_dice) / (1 + self.small_weight)
        
        # Return loss
        return 1 - dice_score 


class EnhancedCombinedLoss(nn.Module):
    """
    Multi-component loss function combining Dice, BCE, and Focal losses.
    
    This loss function addresses multiple challenges in medical image segmentation:
    - Class imbalance (through focal loss and positive weighting)
    - Small object detection (through small tumor-aware Dice loss)
    - Pixel-wise accuracy (through BCE loss)
    
    Attributes:
        dice_weight (float): Weight for Dice loss component (default: 0.5)
        bce_weight (float): Weight for BCE loss component (default: 0.3)
        focal_weight (float): Weight for Focal loss component (default: 0.2)
        small_tumor_weight (float): Weight for small tumor emphasis (default: 1.5)
        pos_weight (float): Positive class weight for BCE (default: 1.3)
    
    """
    def __init__(self, dice_weight=0.5, bce_weight=0.3, focal_weight=0.2, 
                 small_tumor_weight=1.5, pos_weight=1.3):
        """
        Initialize the EnhancedCombinedLoss.
        
        Args:
            dice_weight (float): Weight for Dice loss component
            bce_weight (float): Weight for BCE loss component
            focal_weight (float): Weight for Focal loss component
            small_tumor_weight (float): Emphasis on small tumor detection
            pos_weight (float): Weight for positive class in BCE
        """
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.dice_loss = SmallTumorDiceLoss(smooth=1.0, small_weight=small_tumor_weight)
        self.register_buffer('pos_weight', torch.tensor([pos_weight]))
    
    def _focal_loss(self, pred_logits, target, alpha=0.25, gamma=2.0):
        """
        Compute Focal Loss for addressing class imbalance.
        
        Focal Loss down-weights easy examples and focuses training on hard negatives.
        
        Args:
            pred_logits (torch.Tensor): Raw model outputs (before sigmoid)
            target (torch.Tensor): Ground truth binary masks
            alpha (float): Balancing factor for positive/negative classes (default: 0.25)
            gamma (float): Focusing parameter (default: 2.0). Higher = more focus on hard examples
        
        Returns:
            torch.Tensor: Scalar focal loss value
        """
        # Calculate base binary cross-entropy
        bce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')

        # Calculate probability of correct class
        pt = torch.exp(-bce)

        # Apply focal weight: α(1-pt)^γ
        focal_weight = alpha * (1 - pt) ** gamma

        return (focal_weight * bce).mean()
    
    def forward(self, pred_logits, target):
        """
        Compute the combined loss from all components.
        
        Args:
            pred_logits (torch.Tensor): Raw model outputs [B, 1, H, W] (before sigmoid)
            target (torch.Tensor): Ground truth binary masks [B, 1, H, W]
        
        Returns:
            tuple: (total_loss, components_dict)
        """
        
        # Apply sigmoid for Dice loss (requires probabilities)
        pred_sigmoid = torch.sigmoid(pred_logits)
        dice = self.dice_loss(pred_sigmoid, target)
        bce = F.binary_cross_entropy_with_logits(pred_logits, target, pos_weight=self.pos_weight)
        focal = self._focal_loss(pred_logits, target)
        total = (self.dice_weight * dice + self.bce_weight * bce + self.focal_weight * focal)
        
        return total, {
            'dice': dice.item(),
            'bce': bce.item(),
            'focal': focal.item(),
            'total': total.item()
        }


class TrainingHistory:
    """
    Track and visualize training metrics over epochs.
    
    This class maintains a record of all training metrics and provides
    visualization capabilities for monitoring training progress.
    
    Tracked Metrics:
        - Training loss
        - Validation loss
        - Validation Dice score
        - Validation F1 score
        - Learning rate
    
    Attributes:
        train_losses (list): Training loss per epoch
        val_losses (list): Validation loss per epoch
        val_dice_scores (list): Validation Dice scores per epoch
        val_f1_scores (list): Validation F1 scores per epoch
        learning_rates (list): Learning rate per epoch
        epochs (list): Epoch numbers
    """

    def __init__(self):
        """Initialize empty tracking lists."""
        self.train_losses = []
        self.val_losses = []
        self.val_dice_scores = []
        self.val_f1_scores = []
        self.learning_rates = []
        self.epochs = []
        
    def update(self, epoch, train_loss, val_loss, val_dice, val_f1, lr):
        """
        Record metrics for a single epoch.
        
        Args:
            epoch (int): Current epoch number
            train_loss (float): Average training loss for the epoch
            val_loss (float): Average validation loss for the epoch
            val_dice (float): Validation Dice score
            val_f1 (float): Validation F1 score
            lr (float): Current learning rate
        """
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.val_dice_scores.append(val_dice)
        self.val_f1_scores.append(val_f1)
        self.learning_rates.append(lr)
    
    def plot(self, save_path):
        """
        Generate and save a 2x2 grid of training curve plots.
        
        Creates four subplots:
            1. Training and Validation Loss
            2. Validation Dice Score
            3. Validation F1 Score
            4. Learning Rate Schedule 
        
        Args:
            save_path 
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        axes[0, 0].plot(self.epochs, self.train_losses, label='Train Loss', linewidth=2)
        axes[0, 0].plot(self.epochs, self.val_losses, label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(self.epochs, self.val_dice_scores, label='Dice Score', 
                       linewidth=2, color='green')
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('Dice Score', fontsize=12)
        axes[0, 1].set_title('Validation Dice Score', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0, 1])
        
        axes[1, 0].plot(self.epochs, self.val_f1_scores, label='F1 Score', 
                       linewidth=2, color='orange')
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('F1 Score', fontsize=12)
        axes[1, 0].set_title('Validation F1 Score', fontsize=14, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 1])
        
        axes[1, 1].plot(self.epochs, self.learning_rates, label='Learning Rate', 
                       linewidth=2, color='red')
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Learning Rate', fontsize=12)
        axes[1, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


def setup_training():
    """
    Initialize the training environment and create necessary directories.
    
    This function performs the following setup tasks:
        1. Sets random seeds for reproducibility
        2. Configures PyTorch backend for deterministic behavior
        3. Creates directory structure for saving outputs
    
    Returns:
        tuple: (checkpoint_dir, results_dir, plots_dir)
    """

    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    checkpoint_dir = Path("checkpoints")
    results_dir = Path("results")
    plots_dir = Path("plots")
    
    for d in [checkpoint_dir, results_dir, plots_dir]:
        d.mkdir(exist_ok=True)
    
    return checkpoint_dir, results_dir, plots_dir


class MixedPrecisionTrainer:
    """
    Handle mixed precision training for improved performance and memory efficiency.
    
    Mixed precision training uses float16 (half precision) for most operations
    while maintaining float32 precision for critical calculations.
    
    The trainer automatically handles:
        - Gradient scaling to prevent underflow
        - Loss scaling for stable training
        - Gradient clipping for training stability
    
    Attributes:
        device (torch.device): Device for training (CPU or CUDA)
        scaler (GradScaler or None): Gradient scaler for mixed precision (CUDA only)
    """

    def __init__(self, device):
        """
        Initialize the mixed precision trainer.
        
        Args:
            device (torch.device): Device for training (CPU or CUDA)
        """

        self.device = device
        self.scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    def train_step(self, model, images, masks, criterion, optimizer):
        """
        Execute a single training step with mixed precision.
        
        This method performs:
            1. Forward pass (with autocast if CUDA)
            2. Loss calculation
            3. Backward pass with gradient scaling
            4. Gradient clipping 
            5. Optimizer step
        
        Args:
            model (nn.Module): Neural network model
            images (torch.Tensor): Input images [B, C, H, W]
            masks (torch.Tensor): Ground truth masks [B, 1, H, W]
            criterion (nn.Module): Loss function
            optimizer (torch.optim.Optimizer): Optimizer
        
        Returns:
            tuple: (loss, components, outputs)
        
        Training Flow (CUDA):
            with autocast:
                outputs = model(images)        
                loss = criterion(outputs, masks)
            
            scaler.scale(loss).backward()     
            scaler.unscale_(optimizer)         
            clip_grad_norm_(...)              
            scaler.step(optimizer)            
            scaler.update()                   
        """

        if self.scaler:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss, components = criterion(outputs, masks)
            
            optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            outputs = model(images)
            loss, components = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        return loss, components, outputs


def train_epoch(model, loader, criterion, optimizer, device, trainer):
    """
    Train the model for one complete epoch.
    
    Iterates through all batches in the training set, performing forward and
    backward passes for each batch. Returns average metrics over the epoch.
    
    Args:
        model (nn.Module): Neural network model to train
        loader (DataLoader): Training data loader
        criterion (nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer for weight updates
        device (torch.device): Device for computation (CPU or CUDA)
        trainer (MixedPrecisionTrainer): Mixed precision training handler
    
    Returns:
        tuple: (avg_loss, avg_dice)
            - avg_loss (float): Average training loss over the epoch
            - avg_dice (float): Average Dice loss component over the epoch
    """

    model.train()
    total_loss = 0
    total_dice = 0
    samples = 0
    
    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        loss, components, outputs = trainer.train_step(model, images, masks, criterion, optimizer)
        
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_dice += components['dice'] * batch_size
        samples += batch_size
    
    avg_loss = total_loss / samples if samples > 0 else 0
    avg_dice = total_dice / samples if samples > 0 else 0
    
    return avg_loss, avg_dice


@torch.no_grad()
def validate_epoch(model, loader, criterion, device):
    """
    Validate the model on a validation or test set.
    
    Performs inference on all batches without gradient computation and calculates
    comprehensive evaluation metrics including standard segmentation metrics and
    specialized small tumor detection metrics.
    
    Args:
        model (nn.Module): Neural network model to evaluate
        loader (DataLoader): Validation/test data loader
        criterion (nn.Module): Loss function
        device (torch.device): Device for computation
    
    Returns:
        tuple: (avg_loss, dice, metrics)
            - avg_loss (float): Average validation loss
            - dice (float): Overall Dice coefficient
            - metrics (dict): Comprehensive evaluation metrics
    
    Small Tumor Tracking:
        Tumors with ground truth size in (0, 500) pixels are tracked separately
        to evaluate model performance on small lesions, which are clinically
        important but harder to detect.
    """

    model.eval()
    total_loss = 0
    samples = 0
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    
    small_tumor_tp = 0
    small_tumor_fp = 0
    small_tumor_fn = 0
    small_tumor_count = 0
    small_tumor_total_size = 0
    
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        outputs = model(images)
        loss, components = criterion(outputs, masks)
        
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        samples += batch_size
        
        pred_sigmoid = torch.sigmoid(outputs)
        pred_binary = (pred_sigmoid > 0.5).float()
        
        total_tp += (pred_binary * masks).sum().item()
        total_fp += (pred_binary * (1 - masks)).sum().item()
        total_fn += ((1 - pred_binary) * masks).sum().item()
        total_tn += ((1 - pred_binary) * (1 - masks)).sum().item()
        
        for i in range(batch_size):
            tumor_size = masks[i].sum().item()
            if 0 < tumor_size < 500:
                pred_i = pred_binary[i]
                mask_i = masks[i]
                
                small_tumor_tp += (pred_i * mask_i).sum().item()
                small_tumor_fp += (pred_i * (1 - mask_i)).sum().item()
                small_tumor_fn += ((1 - pred_i) * mask_i).sum().item()
                small_tumor_count += 1
                small_tumor_total_size += tumor_size
    
    avg_loss = total_loss / samples if samples > 0 else 0
    
    eps = 1e-7
    precision = total_tp / (total_tp + total_fp + eps)
    recall = total_tp / (total_tp + total_fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    dice = 2 * total_tp / (2 * total_tp + total_fp + total_fn + eps)
    accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn + eps)
    iou = total_tp / (total_tp + total_fp + total_fn + eps)
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'dice': dice,
        'iou': iou,
        'tp': total_tp,
        'fp': total_fp,
        'tn': total_tn,
        'fn': total_fn
    }
    
    if small_tumor_count > 0:
        small_tumor_dice = 2 * small_tumor_tp / (2 * small_tumor_tp + small_tumor_fp + small_tumor_fn + eps)
        metrics['small_tumor_dice'] = small_tumor_dice
        metrics['small_tumor_count'] = small_tumor_count
        metrics['avg_small_tumor_size'] = small_tumor_total_size / small_tumor_count
    
    return avg_loss, dice, metrics


def main():
    """
    Main training pipeline for medical image segmentation.
    
    Training Pipeline:
        1. Environment setup and directory creation
        2. Data loader initialization with augmentation
        3. Model, loss function, and optimizer instantiation
        4. Training loop with validation after each epoch
        5. Best model checkpointing based on validation Dice score
        6. Periodic checkpointing every 10 epochs
        7. Early stopping if validation doesn't improve
        8. Final evaluation on held-out test set
        9. Results saving (JSON) and training curve visualization (PNG)
    
    Training Features:
        - Mixed precision training for CUDA devices (~2x speedup)
        - AdamW optimizer with weight decay regularization
        - ReduceLROnPlateau scheduler (halves LR after 5 epochs without improvement)
        - Early stopping with 20-epoch patience
        - Gradient clipping (max_norm=1.0) for training stability
        - Comprehensive metrics tracking (Dice, F1, IoU, etc.)
        - Special tracking for small tumors (< 500 pixels)
    
    Hyperparameters:
        Architecture (from config):
            - INPUT_CHANNELS: Number of input image channels 
            - NUM_CLASSES: Output classes (1 for binary segmentation)
            - BASE_FILTERS: Base number of filters in U-Net
            - DROPOUT_RATE: Dropout probability for regularization
        
        Training (from config):
            - BATCH_SIZE: Samples per training batch
            - DICE_WEIGHT: Weight for Dice loss component (default: 0.5)
            - BCE_WEIGHT: Weight for binary cross-entropy (default: 0.3)
            - FOCAL_WEIGHT: Weight for focal loss component (default: 0.2)
            - POS_WEIGHT: Positive class weight for handling imbalance
            - RANDOM_SEED: For reproducibility
        
        Optimization:
            - Learning rate: 1e-4 (initial)
            - Weight decay: 1e-5 (L2 regularization)
            - Betas: (0.9, 0.999) for AdamW momentum
            - LR scheduler: ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
            - Max epochs: 100
            - Early stopping patience: 20 epochs
    
    Early Stopping:
        Training stops if validation Dice score doesn't improve for 20 consecutive
        epochs. The best model (highest validation Dice) is automatically saved
        and used for final test evaluation.
    
    Device Selection:
        Automatically uses CUDA if available, otherwise falls back to CPU.
        Mixed precision training is enabled only for CUDA devices.

    Raises:
        Exception: Any errors during training are caught by the __main__ block
                  and printed with full traceback for debugging
    
    Example Usage:
        >>> # Run training from command line
        >>> python train.py
        
        >>> # Or import and run programmatically
        >>> if __name__ == "__main__":
        >>>     main()
    """

    # Record start time for total training duration
    start_time = time.time()
    
    # Create output directories and set random seeds for reproducibility
    checkpoint_dir, results_dir, plots_dir = setup_training()
    
    # Select compute device (CUDA GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # DATA LOADING
    # Create data loaders for train/validation/test sets
    # use_augmentation=True enables random flips, rotations, etc. for training
    train_loader, val_loader, test_loader = create_dataloaders(
        config, use_augmentation=True
    )
    
    # MODEL INITIALIZATION

    # Initialize Attention U-Net architecture
    model = AttentionUNet(
        in_channels=config.INPUT_CHANNELS,    
        num_classes=config.NUM_CLASSES,        
        base_filters=config.BASE_FILTERS       
    )
    model = model.to(device)
    
    # Count total trainable parameters for reporting
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # LOSS FUNCTION INITIALIZATION

    # Initialize combined loss function with three components:
    # 1. Dice loss (overlap-based, handles class imbalance)
    # 2. BCE loss (pixel-wise binary cross-entropy)
    # 3. Focal loss (focuses on hard examples)
    # small_tumor_weight=1.5 gives extra emphasis to small lesions
    criterion = EnhancedCombinedLoss(
        dice_weight=config.DICE_WEIGHT,        # Weight for Dice component
        bce_weight=config.BCE_WEIGHT,          # Weight for BCE component
        focal_weight=config.FOCAL_WEIGHT,      # Weight for Focal component
        small_tumor_weight=1.5,                # Extra weight for small tumors
        pos_weight=config.POS_WEIGHT           # Positive class weight in BCE
    )
    criterion = criterion.to(device)

    # OPTIMIZER AND SCHEDULER INITIALIZATION
    
    # AdamW optimizer with weight decay for regularization
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-4,              # Initial learning rate
        weight_decay=1e-5,    # L2 regularization strength
        betas=(0.9, 0.999)    # Momentum parameters
    )
    
    # Learning rate scheduler: reduces LR when validation loss plateaus
    # Halves the learning rate if no improvement for 5 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',      # Minimize validation loss
        factor=0.5,      # Multiply LR by 0.5 when triggered
        patience=5,      # Wait 5 epochs before reducing
        min_lr=1e-6      # Don't reduce below this value
    )
    
    # TRAINING UTILITIES INITIALIZATION
    #     
    # Mixed precision trainer for faster training on CUDA
    trainer = MixedPrecisionTrainer(device)
    
    # History tracker for metrics and visualization
    history = TrainingHistory()
    
    # TRAINING CONFIGURATION
    
    num_epochs = 100                    # Maximum training epochs
    best_val_loss = float('inf')        # Track best validation loss
    best_dice = 0                       # Track best validation Dice score
    patience_counter = 0                # Count epochs without improvement
    patience_limit = 20                 # Stop if no improvement for this many epochs
    
    # MAIN TRAINING LOOP
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Train for one epoch
        train_loss, train_dice = train_epoch(
            model, train_loader, criterion, optimizer, device, trainer
        )
        
        # Validate on validation set
        val_loss, val_dice, val_metrics = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record metrics in history for later plotting
        history.update(
            epoch, train_loss, val_loss, val_dice, val_metrics['f1'], current_lr
        )
        
        # MODEL CHECKPOINTING
        
        # Check if this is the best model so far (based on validation Dice)
        is_best = val_dice > best_dice
        
        if is_best:
            # Update best scores and reset patience counter
            best_dice = val_dice
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model checkpoint with full state
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_dice': val_dice,
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'config': {
                    'INPUT_CHANNELS': config.INPUT_CHANNELS,
                    'NUM_CLASSES': config.NUM_CLASSES,
                    'BASE_FILTERS': config.BASE_FILTERS
                }
            }, checkpoint_dir / "best_model.pth")
        else:
            # Increment patience counter if no improvement
            patience_counter += 1
        
        # PROGRESS LOGGING
        
        # Calculate time taken for this epoch
        epoch_time = time.time() - epoch_start
        
        # Format small tumor metrics if available
        small_tumor_info = ""
        if 'small_tumor_dice' in val_metrics:
            small_tumor_info = (
                f" | Small Dice: {val_metrics['small_tumor_dice']:.4f} "
                f"(n={val_metrics['small_tumor_count']})"
            )
        
        # Print epoch summary with BEST marker if this is the best model
        status = "BEST" if is_best else "    "
        print(f"{status} Epoch {epoch:03d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Dice: {val_dice:.4f} | "
              f"F1: {val_metrics['f1']:.4f}"
              f"{small_tumor_info} | "
              f"Time: {epoch_time:.1f}s | "
              f"LR: {current_lr:.2e}")
        
        # PERIODIC CHECKPOINTING AND PLOTTING
        
        # Every 10 epochs (after epoch 0), save checkpoint and plot curves
        if epoch % 10 == 0 and epoch > 0:
            # Save periodic checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_dice': val_dice
            }, checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pth")
            
            # Generate and save training curves plot
            history.plot(plots_dir / f"training_curves_epoch_{epoch:03d}.png")
        
        # EARLY STOPPING CHECK
        
        # Stop training if no improvement for patience_limit epochs
        if patience_counter >= patience_limit:
            break
    
    # FINAL PLOTTING
    
    # Save final training curves after training completes
    history.plot(plots_dir / "training_curves_final.png")
    
    # Calculate total training time
    total_time = time.time() - start_time
    
    # TEST SET EVALUATION
    
    # Load the best model for final evaluation
    best_checkpoint = torch.load(
        checkpoint_dir / "best_model.pth", map_location=device
    )
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    # Evaluate on held-out test set
    test_loss, test_dice, test_metrics = validate_epoch(
        model, test_loader, criterion, device
    )
    
    # RESULTS PRINTING
    
    # Print formatted test results table
    print(f"\n{'Metric':<20} {'Score':>10}")
    print(f"{'Loss':<20} {test_loss:>10.4f}")
    print(f"{'Dice Score':<20} {test_dice:>10.4f}")
    print(f"{'F1 Score':<20} {test_metrics['f1']:>10.4f}")
    print(f"{'IoU':<20} {test_metrics['iou']:>10.4f}")
    print(f"{'Precision':<20} {test_metrics['precision']:>10.4f}")
    print(f"{'Recall':<20} {test_metrics['recall']:>10.4f}")
    print(f"{'Accuracy':<20} {test_metrics['accuracy']:>10.4f}")
    
    # Print small tumor metrics if available
    if 'small_tumor_dice' in test_metrics:
        print("-" * 32)
        print(f"{'Small Tumor Dice':<20} {test_metrics['small_tumor_dice']:>10.4f}")
        print(f"{'Small Tumor Count':<20} {test_metrics['small_tumor_count']:>10}")
        print(f"{'Avg Small Size (px)':<20} {test_metrics['avg_small_tumor_size']:>10.1f}")
    
    # RESULTS SAVING
    
    # Compile comprehensive results dictionary
    results = {
        'test_loss': float(test_loss),
        'test_dice': float(test_dice),
        'test_metrics': {
            k: float(v) if isinstance(v, (int, float, np.number)) else v 
            for k, v in test_metrics.items()
        },
        'best_val_dice': float(best_dice),
        'total_epochs': epoch + 1,
        'training_time_minutes': total_time / 60,
        'model_parameters': total_params,
        'training_config': {
            'batch_size': config.BATCH_SIZE,
            'base_filters': config.BASE_FILTERS,
            'dropout_rate': config.DROPOUT_RATE,
            'random_seed': config.RANDOM_SEED
        }
    }
    
    # Save results to JSON file for reproducibility and record-keeping
    results_path = results_dir / "final_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    try:
        # Run main training pipeline
        main()
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        print("\nInterrupted")
    except Exception as e:
        # Catch and print any errors with full traceback for debugging
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()