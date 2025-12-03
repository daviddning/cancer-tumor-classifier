## 3-stage training script for brain tumor classification ##

import os
import sys
from pathlib import Path
import time
import copy
from typing import Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torchvision import models
from torchvision.models import ResNet50_Weights
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    CLASSES, NUM_CLASSES, DEVICE, CHECKPOINT_DIR, GRAPHICS_DIR, GRADCAM_DIR,
    STAGE1_EPOCHS, STAGE2_EPOCHS, STAGE3_EPOCHS,
    LR_STAGE1, LR_STAGE2, LR_STAGE3, WEIGHT_DECAY,
    PATIENCE_STAGE2, PATIENCE_STAGE3, MIN_DELTA_STAGE2, MIN_DELTA_STAGE3,
    DROPOUT_RATE, GRADIENT_CLIP, LABEL_SMOOTHING, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, PERSISTENT_WORKERS,
    STAGE1_T0, STAGE1_T_MULT, STAGE2_LR_FACTOR, STAGE2_SCHEDULER_PATIENCE, STAGE3_LR_FACTOR, STAGE3_SCHEDULER_PATIENCE,
    PRETRAINED_WEIGHTS_PATH, GRADCAM_CONFIG
)
from data_processing import process_and_prepare_data, save_augmented_samples
from gradcam_utils import create_gradcam_visualization



class BrainTumorClassifier(nn.Module):
    
    ## resnet50-based classifier with custom head for brain tumor classification ##

    def __init__(self, num_classes=NUM_CLASSES, dropout_rate=DROPOUT_RATE, pretrained=True):
        super(BrainTumorClassifier, self).__init__()
        
        # load pre-trained resnet50
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        
        # load custom weights if provided
        if PRETRAINED_WEIGHTS_PATH is not None:
            checkpoint = torch.load(PRETRAINED_WEIGHTS_PATH, map_location=DEVICE)
            self.backbone.load_state_dict(checkpoint)
        
        # get feature dimension
        num_features = self.backbone.fc.in_features
        
        # replace classifier head with custom architecture
        self.backbone.fc = nn.Identity()
        
        # custom classification head with dropout and batch norm
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output


class EarlyStopping:
    
    ## early stopping to prevent overfitting ##

    def __init__(self, patience=PATIENCE_STAGE2, min_delta=MIN_DELTA_STAGE2, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, score, epoch):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        if self.mode == 'max':
            improved = score > (self.best_score + self.min_delta)
        else:
            improved = score < (self.best_score - self.min_delta)
        
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
        return self.early_stop


class MetricsTracker:
    
    ## track and store training metrics ##

    def __init__(self):
        self.history = {
            'train_loss': [], 'train_acc': [], 'train_f1': [],
            'val_loss': [], 'val_acc': [], 'val_f1': [],
            'lr': []
        }
        
    ## update metrics for given phase ##
    def update(self, metrics: Dict, phase: str):
        for key, value in metrics.items():
            history_key = f"{phase}_{key}"
            if history_key in self.history:
                self.history[history_key].append(value)
    
    ## plot training metrics ##
    def plot_metrics(self, save_path: Path, stage_name: str):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{stage_name} Training Metrics', fontsize=16, fontweight='bold')
        
        # loss
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss', linewidth=2)
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss over Epochs')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # accuracy
        axes[0, 1].plot(self.history['train_acc'], label='Train Acc', linewidth=2)
        axes[0, 1].plot(self.history['val_acc'], label='Val Acc', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Accuracy over Epochs')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # f1 score
        axes[1, 0].plot(self.history['train_f1'], label='Train F1', linewidth=2)
        axes[1, 0].plot(self.history['val_f1'], label='Val F1', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].set_title('F1 Score over Epochs')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # learning rate
        axes[1, 1].plot(self.history['lr'], linewidth=2, color='green')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Metrics plot saved to {save_path}")


def calculate_metrics(outputs: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:

    ## calculate classification metrics ##

    _, preds = torch.max(outputs, 1)
    
    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    metrics = {
        'acc': accuracy_score(labels_np, preds_np),
        'f1': f1_score(labels_np, preds_np, average='weighted', zero_division=0),
        'precision': precision_score(labels_np, preds_np, average='weighted', zero_division=0),
        'recall': recall_score(labels_np, preds_np, average='weighted', zero_division=0)
    }
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, save_path: Path, title: str):
    cm = confusion_matrix(y_true, y_pred)
    
    ## plot and save confusion matrix ##

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASSES, yticklabels=CLASSES,
                cbar_kws={'label': 'Count'})
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to {save_path}")


def train_epoch(model, dataloader, criterion, optimizer, device, clip_grad=GRADIENT_CLIP):
    ## train for one epoch ##
    
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc='Training', leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        
        # gradient clipping
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return {
        'loss': epoch_loss,
        'acc': epoch_acc,
        'f1': epoch_f1
    }


def validate_epoch(model, dataloader, criterion, device):
    ## validate for one epoch ##
    
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation', leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return {
        'loss': epoch_loss,
        'acc': epoch_acc,
        'f1': epoch_f1
    }, all_labels, all_preds


def save_checkpoint(model, optimizer, epoch, metrics, stage, save_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'stage': stage
    }

    ## save model checkpoint ##

    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved: {save_path}")


## stage 1: freeze backbone, train only classifier head, fast convergence on new task ##
def stage1_training(model, dataloaders, device):
    print("\n" + "="*80)
    print("STAGE 1: Training Classifier Head (Frozen Backbone)")
    print("="*80)
    
    # freeze backbone
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    # only train classifier
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.classifier.parameters(), lr=LR_STAGE1, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=STAGE1_T0, T_mult=STAGE1_T_MULT)
    
    metrics_tracker = MetricsTracker()
    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    for epoch in range(STAGE1_EPOCHS):
        print(f"\nEpoch {epoch+1}/{STAGE1_EPOCHS}")
        print("-" * 40)
        
        # train
        train_metrics = train_epoch(model, dataloaders['train'], criterion, optimizer, device)
        
        # validate
        val_metrics, val_labels, val_preds = validate_epoch(model, dataloaders['val'], criterion, device)
        
        # re-enable training mode for next epoch
        model.train()
        
        # update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # track metrics
        metrics_tracker.update(train_metrics, 'train')
        metrics_tracker.update(val_metrics, 'val')
        metrics_tracker.history['lr'].append(current_lr)
        
        # print epoch summary
        print(f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['acc']:.4f} | Train F1: {train_metrics['f1']:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['acc']:.4f} | Val F1: {val_metrics['f1']:.4f}")
        print(f"LR: {current_lr:.6f}")
        
        # save best model
        if val_metrics['acc'] > best_val_acc:
            best_val_acc = val_metrics['acc']
            best_model_wts = copy.deepcopy(model.state_dict())
            save_checkpoint(
                model, optimizer, epoch, val_metrics, 'stage1',
                CHECKPOINT_DIR / 'stage1_best.pth'
            )
    
    # load best weights
    model.load_state_dict(best_model_wts)
    
    # gradcam visualization
    if GRADCAM_CONFIG['checkpoints']['stage1_end']:
        create_gradcam_visualization(
            model, dataloaders['val'], 'stage1', 
            STAGE1_EPOCHS - 1,
            num_samples=GRADCAM_CONFIG['samples_per_class'],
            save_dir=GRADCAM_DIR
        )
    
    # plot metrics
    metrics_tracker.plot_metrics(GRAPHICS_DIR / 'stage1_metrics.png', 'Stage 1')
    
    # confusion matrix
    plot_confusion_matrix(
        val_labels, val_preds,
        GRAPHICS_DIR / 'stage1_confusion_matrix.png',
        'Stage 1 Validation Confusion Matrix'
    )
    
    print(f"\nStage 1 Complete! Best Val Acc: {best_val_acc:.4f}")
    
    return model, metrics_tracker


## stage 2: unfreeze top layers, fine-tune with lower lr, adapt features to specific task ##
def stage2_training(model, dataloaders, device):
    print("\n" + "="*80)
    print("STAGE 2: Fine-tuning Top Layers")
    print("="*80)
    
    # unfreeze layer4 and classifier
    for param in model.backbone.layer4.parameters():
        param.requires_grad = True
    
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    
    # different learning rates for different parts
    optimizer = optim.AdamW([
        {'params': model.backbone.layer4.parameters(), 'lr': LR_STAGE2 * 0.1},
        {'params': model.classifier.parameters(), 'lr': LR_STAGE2}
    ], weight_decay=WEIGHT_DECAY)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=STAGE2_LR_FACTOR, patience=STAGE2_SCHEDULER_PATIENCE, verbose=True)
    early_stopping = EarlyStopping(patience=PATIENCE_STAGE2, min_delta=MIN_DELTA_STAGE2, mode='max')
    
    metrics_tracker = MetricsTracker()
    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    for epoch in range(STAGE2_EPOCHS):
        print(f"\nEpoch {epoch+1}/{STAGE2_EPOCHS}")
        print("-" * 40)
        
        # train
        train_metrics = train_epoch(model, dataloaders['train'], criterion, optimizer, device)
        
        # validate
        val_metrics, val_labels, val_preds = validate_epoch(model, dataloaders['val'], criterion, device)
        
        # re-enable training mode for next epoch
        model.train()
        
        # update learning rate
        scheduler.step(val_metrics['acc'])
        current_lr = optimizer.param_groups[0]['lr']
        
        # track metrics
        metrics_tracker.update(train_metrics, 'train')
        metrics_tracker.update(val_metrics, 'val')
        metrics_tracker.history['lr'].append(current_lr)
        
        # print epoch summary
        print(f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['acc']:.4f} | Train F1: {train_metrics['f1']:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['acc']:.4f} | Val F1: {val_metrics['f1']:.4f}")
        print(f"LR: {current_lr:.6f}")
        
        # save best model
        if val_metrics['acc'] > best_val_acc:
            best_val_acc = val_metrics['acc']
            best_model_wts = copy.deepcopy(model.state_dict())
            save_checkpoint(
                model, optimizer, epoch, val_metrics, 'stage2',
                CHECKPOINT_DIR / 'stage2_best.pth'
            )
        
        # early stopping
        if early_stopping(val_metrics['acc'], epoch):
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            print(f"Best epoch was {early_stopping.best_epoch+1}")
            break
    
    # load best weights
    model.load_state_dict(best_model_wts)
    
    # gradcam visualization
    if GRADCAM_CONFIG['checkpoints']['stage2_end']:
        create_gradcam_visualization(
            model, dataloaders['val'], 'stage2', 
            epoch,
            num_samples=GRADCAM_CONFIG['samples_per_class'],
            save_dir=GRADCAM_DIR
        )
    
    # plot metrics
    metrics_tracker.plot_metrics(GRAPHICS_DIR / 'stage2_metrics.png', 'Stage 2')
    
    # confusion matrix
    plot_confusion_matrix(
        val_labels, val_preds,
        GRAPHICS_DIR / 'stage2_confusion_matrix.png',
        'Stage 2 Validation Confusion Matrix'
    )
    
    print(f"\nStage 2 Complete! Best Val Acc: {best_val_acc:.4f}")
    
    return model, metrics_tracker


## stage 3: full fine-tuning with very low lr, final optimization for maximum accuracy ##
def stage3_training(model, dataloaders, device):
    print("\n" + "="*80)
    print("STAGE 3: Full Model Fine-tuning")
    print("="*80)
    
    # unfreeze all layers
    for param in model.parameters():
        param.requires_grad = True
    
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    
    # different learning rates for different parts
    optimizer = optim.AdamW([
        {'params': model.backbone.layer1.parameters(), 'lr': LR_STAGE3 * 0.01},
        {'params': model.backbone.layer2.parameters(), 'lr': LR_STAGE3 * 0.05},
        {'params': model.backbone.layer3.parameters(), 'lr': LR_STAGE3 * 0.1},
        {'params': model.backbone.layer4.parameters(), 'lr': LR_STAGE3 * 0.5},
        {'params': model.classifier.parameters(), 'lr': LR_STAGE3}
    ], weight_decay=WEIGHT_DECAY)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=STAGE3_LR_FACTOR, patience=STAGE3_SCHEDULER_PATIENCE, verbose=True)
    early_stopping = EarlyStopping(patience=PATIENCE_STAGE3, min_delta=MIN_DELTA_STAGE3, mode='max')
    
    metrics_tracker = MetricsTracker()
    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    for epoch in range(STAGE3_EPOCHS):
        print(f"\nEpoch {epoch+1}/{STAGE3_EPOCHS}")
        print("-" * 40)
        
        # train
        train_metrics = train_epoch(model, dataloaders['train'], criterion, optimizer, device)
        
        # validate
        val_metrics, val_labels, val_preds = validate_epoch(model, dataloaders['val'], criterion, device)
        
        # re-enable training mode for next epoch
        model.train()
        
        # update learning rate
        scheduler.step(val_metrics['acc'])
        current_lr = optimizer.param_groups[0]['lr']
        
        # track metrics
        metrics_tracker.update(train_metrics, 'train')
        metrics_tracker.update(val_metrics, 'val')
        metrics_tracker.history['lr'].append(current_lr)
        
        # print epoch summary
        print(f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['acc']:.4f} | Train F1: {train_metrics['f1']:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['acc']:.4f} | Val F1: {val_metrics['f1']:.4f}")
        print(f"LR: {current_lr:.6f}")
        
        # save best model
        if val_metrics['acc'] > best_val_acc:
            best_val_acc = val_metrics['acc']
            best_model_wts = copy.deepcopy(model.state_dict())
            save_checkpoint(
                model, optimizer, epoch, val_metrics, 'stage3',
                CHECKPOINT_DIR / 'stage3_best.pth'
            )
        
        # save periodic checkpoints
        if (epoch + 1) % 5 == 0:
            save_checkpoint(
                model, optimizer, epoch, val_metrics, 'stage3',
                CHECKPOINT_DIR / f'stage3_epoch_{epoch+1}.pth'
            )
        
        # gradcam visualization at intervals
        if GRADCAM_CONFIG['checkpoints']['stage3_interval'] > 0:
            if (epoch + 1) % GRADCAM_CONFIG['checkpoints']['stage3_interval'] == 0:
                create_gradcam_visualization(
                    model, dataloaders['val'], 'stage3', 
                    epoch,
                    num_samples=GRADCAM_CONFIG['samples_per_class'],
                    save_dir=GRADCAM_DIR
                )
        
        # early stopping
        if early_stopping(val_metrics['acc'], epoch):
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            print(f"Best epoch was {early_stopping.best_epoch+1}")
            break
    
    # load best weights
    model.load_state_dict(best_model_wts)
    
    # gradcam visualization
    if GRADCAM_CONFIG['checkpoints']['stage3_start']:
        create_gradcam_visualization(
            model, dataloaders['val'], 'stage3', 
            epoch,
            num_samples=GRADCAM_CONFIG['samples_per_class'],
            save_dir=GRADCAM_DIR
        )
    
    # plot metrics
    metrics_tracker.plot_metrics(GRAPHICS_DIR / 'stage3_metrics.png', 'Stage 3')
    
    # confusion matrix
    plot_confusion_matrix(
        val_labels, val_preds,
        GRAPHICS_DIR / 'stage3_confusion_matrix.png',
        'Stage 3 Validation Confusion Matrix'
    )
    
    print(f"\nStage 3 Complete! Best Val Acc: {best_val_acc:.4f}")
    
    return model, metrics_tracker


## final evaluation on test set ##
def evaluate_final_model(model, dataloader, device):
    print("\n" + "="*80)
    print("FINAL EVALUATION ON TEST SET")
    print("="*80)
    
    # evaluate in eval mode
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Testing')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # calculate metrics
    test_acc = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    test_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    test_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    
    # per-class metrics
    print("\nPer-Class Metrics:")
    for i, class_name in enumerate(CLASSES):
        class_mask = np.array(all_labels) == i
        if class_mask.sum() > 0:
            class_preds = np.array(all_preds)[class_mask]
            class_labels = np.array(all_labels)[class_mask]
            class_acc = accuracy_score(class_labels, class_preds)
            print(f"  {class_name}: {class_acc:.4f}")
    
    # final confusion matrix
    plot_confusion_matrix(
        all_labels, all_preds,
        GRAPHICS_DIR / 'final_test_confusion_matrix.png',
        'Final Test Set Confusion Matrix'
    )
    
    return {
        'accuracy': test_acc,
        'f1': test_f1,
        'precision': test_precision,
        'recall': test_recall
    }


## main training pipeline ##
def main():
    print("="*80)
    print("BRAIN TUMOR CLASSIFICATION - 3-STAGE TRAINING")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"Classes: {CLASSES}")
    print(f"Number of Classes: {NUM_CLASSES}")
    
    # create directories
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHICS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    dataloaders, datasets = process_and_prepare_data()
    
    # Save augmented samples
    print("\nSaving augmented training samples...")
    save_augmented_samples(datasets['train'])
    
    # Initialize model
    print("\n" + "="*80)
    print("INITIALIZING MODEL")
    print("="*80)
    model = BrainTumorClassifier(num_classes=NUM_CLASSES, dropout_rate=DROPOUT_RATE, pretrained=True)
    model = model.to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    
    # Stage 1: Train classifier head
    model, stage1_tracker = stage1_training(model, dataloaders, DEVICE)
    
    # Stage 2: Fine-tune top layers
    model, stage2_tracker = stage2_training(model, dataloaders, DEVICE)
    
    # Stage 3: Full fine-tuning
    model, stage3_tracker = stage3_training(model, dataloaders, DEVICE)
    
    # Final evaluation
    test_metrics = evaluate_final_model(model, dataloaders['test'], DEVICE)
    
    # Save final model
    final_save_path = CHECKPOINT_DIR / 'final_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'test_metrics': test_metrics,
        'classes': CLASSES
    }, final_save_path)
    print(f"\nFinal model saved to {final_save_path}")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nFinal Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Final Test F1 Score: {test_metrics['f1']:.4f}")
    print("\nAll checkpoints and visualizations saved to:")
    print(f"  Checkpoints: {CHECKPOINT_DIR}")
    print(f"  Graphics: {GRAPHICS_DIR}")


if __name__ == "__main__":
    main()