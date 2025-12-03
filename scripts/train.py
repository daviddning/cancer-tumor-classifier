## 3-stage training script for brain tumor classification ##
import os
import sys
from pathlib import Path
import time
import copy
from typing import Dict, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torchvision import models
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from config import *
from data_processing import process_and_prepare_data, save_augmented_samples
from gradcam_utils import create_gradcam_visualization


## resnet50-based classifier with custom head ##
class BrainTumorClassifier(nn.Module):

    def __init__(self, num_classes=NUM_CLASSES, dropout_rate=DROPOUT_RATE):
        super().__init__()
        self.backbone = models.resnet50(weights=None)  
        
        # load radimagenet weights if provided
        if PRETRAINED_WEIGHTS_PATH:
            ckpt = torch.load(PRETRAINED_WEIGHTS_PATH, map_location='cpu')
            if isinstance(ckpt, nn.Module):
                state_dict = ckpt.state_dict()
            elif isinstance(ckpt, dict):
                state_dict = ckpt.get('state_dict', ckpt)
            else:
                raise ValueError("unsupported checkpoint format")
            # remove final classifier
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc')}
            self.backbone.load_state_dict(state_dict, strict=False)
        
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # custom head for 3-class prediction
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
        return self.classifier(self.backbone(x))


## early stopping to halt training when validation metric stops improving ##
class EarlyStopping:

    def __init__(self, patience: int, min_delta: float = 0.0, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'max':
            improved = score > (self.best_score + self.min_delta)
        else:
            improved = score < (self.best_score - self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


## tracks training and validation metrics over epochs ##
class MetricsTracker:

    def __init__(self):
        self.history = {
            'train_loss': [], 'train_acc': [], 'train_f1': [], 'train_precision': [], 'train_recall': [],
            'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_precision': [], 'val_recall': [],
            'lr': []
        }

    def update(self, train_m, val_m, lr):
        self.history['train_loss'].append(train_m['loss'])
        self.history['train_acc'].append(train_m['acc'])
        self.history['train_f1'].append(train_m['f1'])
        self.history['train_precision'].append(train_m['precision'])
        self.history['train_recall'].append(train_m['recall'])
        
        self.history['val_loss'].append(val_m['loss'])
        self.history['val_acc'].append(val_m['acc'])
        self.history['val_f1'].append(val_m['f1'])
        self.history['val_precision'].append(val_m['precision'])
        self.history['val_recall'].append(val_m['recall'])
        
        self.history['lr'].append(lr)

    def plot(self, save_path, title):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # define plot groups: (train_key, val_key, title)
        plot_groups = [
            ('train_loss', 'val_loss', 'loss'),
            ('train_acc', 'val_acc', 'accuracy'),
            ('train_f1', 'val_f1', 'f1 score'),
            ('lr', None, 'learning rate')
        ]
        
        for idx, (train_key, val_key, plot_title) in enumerate(plot_groups):
            ax = axes.flat[idx]
            if val_key is not None:
                ax.plot(self.history[train_key], label='train', linewidth=2)
                ax.plot(self.history[val_key], label='val', linewidth=2)
            else:
                ax.plot(self.history[train_key], label='learning rate', linewidth=2, color='tab:red')
                ax.set_yscale('log')
            ax.set_title(plot_title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


## compute classification metrics from true and predicted labels ##
def calculate_metrics(y_true, y_pred):
    # inputs can be torch.tensor or np.ndarray of shape (n,)
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    return {
        'acc': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0)
    }


## plot and save confusion matrix ##
def plot_confusion_matrix(y_true, y_pred, save_path, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='blues',
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(title)
    plt.ylabel('true')
    plt.xlabel('predicted')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


## train model for one epoch ##
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    for inputs, labels in tqdm(loader, desc='train', leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        # apply gradient clipping if enabled
        if GRADIENT_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(loader.dataset)
    metrics = calculate_metrics(np.array(all_labels), np.array(all_preds))
    return {'loss': epoch_loss, **metrics}


## validate model for one epoch ##
def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc='val', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(loader.dataset)
    metrics = calculate_metrics(np.array(all_labels), np.array(all_preds))
    return {'loss': epoch_loss, **metrics}, all_labels, all_preds


## save model checkpoint ##
def save_checkpoint(model, optimizer, epoch, metrics, stage, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'stage': stage
    }, path)


## run a single training stage with specified configuration ##
def run_stage(
    model, dataloaders, device, stage_name, epochs, 
    lr_config, scheduler_config, early_stop_config,
    gradcam_interval=0
):
    print(f"\n{stage_name.upper()}")
    
    if 'stage1' in stage_name:
        # freeze backbone, train only classifier
        for p in model.backbone.parameters():
            p.requires_grad = False
        for p in model.classifier.parameters():
            p.requires_grad = True
        params = model.classifier.parameters()
        optimizer = optim.AdamW(params, lr=lr_config['lr'], weight_decay=WEIGHT_DECAY)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=STAGE1_T0, T_mult=STAGE1_T_MULT)
        use_scheduler_step = True
        use_early_stopping = False
    else:
        if 'stage2' in stage_name:
            # freeze all but layer4 and classifier
            for p in model.backbone.parameters():
                p.requires_grad = False
            for p in model.backbone.layer4.parameters():
                p.requires_grad = True
            param_groups = [
                {'params': model.backbone.layer4.parameters(), 'lr': lr_config['backbone_lr']},
                {'params': model.classifier.parameters(), 'lr': lr_config['head_lr']}
            ]
        else:  # stage3
            # unfreeze all, use differential learning rates
            for p in model.backbone.parameters():
                p.requires_grad = True
            param_groups = [
                {'params': model.backbone.layer1.parameters(), 'lr': lr_config['l1']},
                {'params': model.backbone.layer2.parameters(), 'lr': lr_config['l2']},
                {'params': model.backbone.layer3.parameters(), 'lr': lr_config['l3']},
                {'params': model.backbone.layer4.parameters(), 'lr': lr_config['l4']},
                {'params': model.classifier.parameters(),     'lr': lr_config['head']}
            ]
        optimizer = optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
        scheduler = ReduceLROnPlateau(
            optimizer, mode='max', 
            factor=scheduler_config['factor'], 
            patience=scheduler_config['patience'], 
            verbose=True
        )
        use_scheduler_step = False
        use_early_stopping = True

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    tracker = MetricsTracker()
    best_acc, best_wts = 0.0, copy.deepcopy(model.state_dict())
    val_labels, val_preds = [], []
    last_epoch = 0

    # initialize early stopping if needed
    if use_early_stopping:
        early_stopping = EarlyStopping(
            patience=early_stop_config['patience'],
            min_delta=early_stop_config['min_delta'],
            mode='max'
        )

    for epoch in range(epochs):
        last_epoch = epoch
        print(f"\nepoch {epoch+1}/{epochs}")
        train_metrics = train_epoch(model, dataloaders['train'], criterion, optimizer, device)
        val_metrics, val_labels, val_preds = validate_epoch(model, dataloaders['val'], criterion, device)
        model.train()

        current_lr = optimizer.param_groups[0]['lr']
        if use_scheduler_step:
            scheduler.step()
        else:
            scheduler.step(val_metrics['acc'])

        tracker.update(train_metrics, val_metrics, current_lr)
        print(f"train: loss={train_metrics['loss']:.4f}, acc={train_metrics['acc']:.4f}")
        print(f"val:   loss={val_metrics['loss']:.4f}, acc={val_metrics['acc']:.4f}, lr={current_lr:.2e}")

        # save best model based on validation accuracy
        if val_metrics['acc'] > best_acc:
            best_acc = val_metrics['acc']
            best_wts = copy.deepcopy(model.state_dict())
            save_checkpoint(model, optimizer, epoch, val_metrics, stage_name, CHECKPOINT_DIR / f"{stage_name}_best.pth")

        # periodic grad-cam visualizations during stage 3
        if 'stage3' in stage_name and gradcam_interval > 0 and (epoch + 1) % gradcam_interval == 0:
            create_gradcam_visualization(
                model, dataloaders['val'], stage_name, epoch,
                GRADCAM_CONFIG['samples_per_class'], GRADCAM_DIR
            )

        # check for early stopping
        if use_early_stopping and early_stopping(val_metrics['acc']):
            print(f"early stopping at epoch {epoch+1}")
            break

    # load best weights
    model.load_state_dict(best_wts)
    
    # final grad-cam after stage 3
    if 'stage3' in stage_name:
        create_gradcam_visualization(
            model, dataloaders['val'], stage_name, last_epoch,
            GRADCAM_CONFIG['samples_per_class'], GRADCAM_DIR
        )

    # save metrics and confusion matrix
    tracker.plot(GRAPHICS_DIR / f"{stage_name}_metrics.png", f"{stage_name.title()} metrics")
    plot_confusion_matrix(
        val_labels, val_preds, 
        GRAPHICS_DIR / f"{stage_name}_confusion.png",
        f"{stage_name.title()} confusion matrix"
    )
    print(f"\n{stage_name} complete. best validation accuracy: {best_acc:.4f}")
    return model, tracker


## evaluate final model on test set ##
def evaluate_final_model(model, dataloader, device):
    print(f"\nfinal evaluation")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='test'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    metrics = calculate_metrics(np.array(all_labels), np.array(all_preds))
    print(f"test accuracy: {metrics['acc']:.4f}, f1: {metrics['f1']:.4f}")
    plot_confusion_matrix(all_labels, all_preds, GRAPHICS_DIR / 'final_confusion.png', 'final test confusion matrix')
    return metrics


## main training pipeline ##
def main():
    print(f"\nbrain tumor classification")
    print(f"device: {DEVICE}, classes: {CLASSES}")
    
    # create required directories
    for d in [CHECKPOINT_DIR, GRAPHICS_DIR, GRADCAM_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    
    # prepare data
    dataloaders, datasets = process_and_prepare_data()
    save_augmented_samples(datasets['train'])
    
    # initialize model
    model = BrainTumorClassifier(NUM_CLASSES, DROPOUT_RATE).to(DEVICE)

    # stage 1: train classifier head only
    model, _ = run_stage(
        model, dataloaders, DEVICE, 'stage1', STAGE1_EPOCHS,
        lr_config={'lr': LR_STAGE1},
        scheduler_config={},
        early_stop_config={}
    )

    # stage 2: fine-tune top layers
    model, _ = run_stage(
        model, dataloaders, DEVICE, 'stage2', STAGE2_EPOCHS,
        lr_config={'backbone_lr': LR_STAGE2 * 0.1, 'head_lr': LR_STAGE2},
        scheduler_config={'factor': STAGE2_LR_FACTOR, 'patience': STAGE2_SCHEDULER_PATIENCE},
        early_stop_config={'patience': PATIENCE_STAGE2, 'min_delta': MIN_DELTA_STAGE2}
    )

    # stage 3: full fine-tuning with differential learning rates
    model, _ = run_stage(
        model, dataloaders, DEVICE, 'stage3', STAGE3_EPOCHS,
        lr_config={
            'l1': LR_STAGE3 * 0.01, 'l2': LR_STAGE3 * 0.02,
            'l3': LR_STAGE3 * 0.05, 'l4': LR_STAGE3 * 0.1,
            'head': LR_STAGE3
        },
        scheduler_config={'factor': STAGE3_LR_FACTOR, 'patience': STAGE3_SCHEDULER_PATIENCE},
        early_stop_config={'patience': PATIENCE_STAGE3, 'min_delta': MIN_DELTA_STAGE3},
        gradcam_interval=GRADCAM_CONFIG['checkpoints']['stage3_interval']
    )

    # final evaluation
    test_metrics = evaluate_final_model(model, dataloaders['test'], DEVICE)
    torch.save({
        'model_state_dict': model.state_dict(), 
        'test_metrics': test_metrics, 
        'classes': CLASSES
    }, CHECKPOINT_DIR / 'final_model.pth')
    print(f"\ntraining complete; final accuracy: {test_metrics['acc']:.4f}")


if __name__ == "__main__":
    main()