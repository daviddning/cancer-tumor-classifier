import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path
import time
import random
import numpy as np
from config import *
from model import get_efficientnet_v2_small, unfreeze_last_n_blocks
from reports import calculate_accuracy, evaluate_model, plot_training_curves, show_sample_predictions

from torch.utils.data import Dataset, DataLoader
import torch
from pathlib import Path
from config import CLASSES

class TensorDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.data = []
        self.labels = []

        print(f"preloading data from {self.root_dir}...")
        start_time = time.time()  

        for label, cls in enumerate(CLASSES):
            cls_dir = self.root_dir / cls
            if not cls_dir.exists():
                raise FileNotFoundError(f"missing class directory: {cls_dir}")
            for file_path in sorted(cls_dir.glob("*.pt")):
                tensor = torch.load(file_path)
                self.data.append(tensor)
                self.labels.append(label)

        self.data = torch.stack(self.data)
        self.labels = torch.tensor(self.labels)
        end_time = time.time()  

        preload_duration = end_time - start_time
        print(f"loaded {len(self.data)} samples into memory.")
        print(f"preloading time: {preload_duration:.2f}s")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    

def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0
    all_preds = []
    all_labels = []
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_acc += calculate_accuracy(outputs, labels) * batch_size
        total_samples += batch_size
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / total_samples
    avg_acc = total_acc / total_samples
    return avg_loss, avg_acc, all_preds, all_labels

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_acc += calculate_accuracy(outputs, labels) * batch_size
            total_samples += batch_size
            preds = torch.argmax(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / total_samples
    avg_acc = total_acc / total_samples
    return avg_loss, avg_acc, all_preds, all_labels

if __name__ == "__main__":
    start_time = time.time()
    # load datasets
    train_dataset = TensorDataset(TRAIN_DIR)
    val_dataset = TensorDataset(VAL_DIR)
    test_dataset = TensorDataset(TEST_DIR)
    num_workers = 0
    train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
    print(f"train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}")

    # build model 
    model = get_efficientnet_v2_small(pretrained=True).to(DEVICE)

    # stage 1 train only classifier head
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_STAGE1, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    scaler = torch.amp.GradScaler("cuda")
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    print("stage 1 training (classifier head only)")
    for epoch in range(STAGE1_EPOCHS):
        epoch_start = time.time()
        train_loss, train_acc, _, _ = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, scaler)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, DEVICE)
        scheduler.step(val_loss)
        epoch_time = time.time() - epoch_start
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        print(f"epoch {epoch+1}/{STAGE1_EPOCHS} ## "
              f"train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} ## "
              f"val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} ## "
              f"epoch Time: {epoch_time:.2f}s")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"new best model saved with val acc: {val_acc:.4f}")

    # stage 2 fine-tuning
    print("stage 2 fine-tuning")
    unfreeze_last_n_blocks(model, n=3)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_STAGE2, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    for epoch in range(STAGE2_EPOCHS):
        epoch_start = time.time()
        train_loss, train_acc, _, _ = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, scaler)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, DEVICE)
        scheduler.step(val_loss)
        epoch_time = time.time() - epoch_start
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        print(f"epoch {epoch+1}/{STAGE2_EPOCHS} ## "
              f"train loss: {train_loss:.4f}, Acc: {train_acc:.4f} ## "
              f"val loss: {val_loss:.4f}, Acc: {val_acc:.4f} ## "
              f"epoch time: {epoch_time:.2f}s")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"new best model saved with val acc: {val_acc:.4f}")

    # final evaluation on test set
    print("final evaluation on test set")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    evaluate_model(model, test_loader, DEVICE)

    # plot training curves
    plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies)

    # show sample predictions
    show_sample_predictions(model, test_loader)

    end_time = time.time()
    total_time = (end_time - start_time) / 60.0
    print(f"total training time: {total_time:.2f} minutes")