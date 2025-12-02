## data processing pipeline for brain tumor classification ##
## data loading, augmentation, train/val/test splitting, and dataset preparation ##

import os
import shutil
from pathlib import Path
from typing import Tuple, Dict, List
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from collections import Counter

# import config
from config import (
    RAW_DIR, PROCESSED_DIR, TRAIN_DIR, TEST_DIR,
    CLASSES, NUM_CLASSES, BATCH_SIZE, NUM_WORKERS,
    PIN_MEMORY, PERSISTENT_WORKERS, IMAGE_SIZE,
    MEAN, STD, DEVICE, GRAPHICS_DIR
)


class BrainTumorDataset(Dataset):

    ## custom dataset for brain tumor MRI images ##
    
    def __init__(self, image_paths: List[Path], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # load img
        image = Image.open(img_path).convert('RGB')
        
        # transformations
        if self.transform:
            image = self.transform(image)
            
        return image, label


def get_train_transforms():
    
    ## training data transformation for variance ##
    
    return transforms.Compose([
       
        transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
        transforms.RandomCrop(IMAGE_SIZE),
        
        
        transforms.RandomRotation(degrees=15),  
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),  
            scale=(0.9, 1.1),      
        ),
        transforms.RandomHorizontalFlip(p=0.5),  
        
        # intensity augmentations 
        transforms.ColorJitter(
            brightness=0.2, 
            contrast=0.2,    
            saturation=0.1,  
            hue=0.05       
        ),
        transforms.RandomGrayscale(p=0.1),  # randomness
        
        # blur
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
        ], p=0.3),
        
        # convert to tensor and normalize (no erasing)
        transforms.ToTensor(),
        
        # normalize using RadImageNet stats
        transforms.Normalize(mean=MEAN, std=STD)
    ])


def get_val_test_transforms():

    ## minimal transformations for val and test data (only resizing and normalizing) ##    

    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])


def get_tta_transforms():
    ## test time augmentation for confidence##
    base_transform = [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ]
    
    tta_transforms = [
        transforms.Compose(base_transform),
        
        # horizontal flip
        transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ]),
        
        # rotation +5 degrees
        transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomRotation(degrees=(5, 5)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ]),
        
        # rotation -5 degrees
        transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomRotation(degrees=(-5, -5)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ]),
        
        # brightness adjustment
        transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ColorJitter(brightness=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ]),
    ]
    
    return tta_transforms


def collect_data_paths(data_dir: Path) -> Tuple[List[Path], List[int]]:

    ## collects data path of samples ##

    image_paths = []
    labels = []
    
    for class_idx, class_name in enumerate(CLASSES):
        class_dir = data_dir / class_name  
            
        # collect all image files
        for img_path in class_dir.glob('*'):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                image_paths.append(img_path)
                labels.append(class_idx)
    
    return image_paths, labels


def analyze_dataset(labels: List[int], split_name: str = "Dataset"):
    ## see dataset stats ##
    counter = Counter(labels)
    total = len(labels)
    
    print(f"\n{split_name} stats:")
    print(f"total samples: {total}")
    print(f"class distribution:")
    for class_idx, class_name in enumerate(CLASSES):
        count = counter[class_idx]
        percentage = (count / total) * 100
        print(f"  {class_name}: {count} ({percentage:.2f}%)")
    
    return counter


def create_splits(
    image_paths: List[Path],
    labels: List[int],
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 88
) -> Dict[str, Tuple[List[Path], List[int]]]:
    
    ## creats train/val/test splits ##
    
    # test set
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths,
        labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state
    )
    
    # validation
    val_size_adjusted = val_size / (1 - test_size)  
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths,
        train_val_labels,
        test_size=val_size_adjusted,
        stratify=train_val_labels,
        random_state=random_state
    )
    
    splits = {
        'train': (train_paths, train_labels),
        'val': (val_paths, val_labels),
        'test': (test_paths, test_labels)
    }
    
    # print split stats
    print("\ndata splits")
    analyze_dataset(train_labels, "training set")
    analyze_dataset(val_labels, "validation set")
    analyze_dataset(test_labels, "test set")
    
    return splits


def save_augmented_samples(
    dataset: BrainTumorDataset,
    num_samples_per_class: int = 3,
    graphics_dir: Path = GRAPHICS_DIR
):
    ## save 3 augmented samples from each class ##
    
    # create graphics directory if it doesn't exist
    graphics_dir.mkdir(parents=True, exist_ok=True)
    
    # get indices for each class
    class_indices = {class_idx: [] for class_idx in range(NUM_CLASSES)}
    for idx, label in enumerate(dataset.labels):
        class_indices[label].append(idx)
    
    print(f"\nsaving augmented samples to {graphics_dir}")
    
    for class_idx, class_name in enumerate(CLASSES):
        if len(class_indices[class_idx]) == 0:
            print(f"  warning: no samples found for class {class_name}")
            continue
        
        # use first sample from this class
        sample_idx = class_indices[class_idx][0]
        
        for i in range(num_samples_per_class):
            # get augmented image (different each time)
            aug_img, _ = dataset[sample_idx]
            
            # denormalize for saving
            aug_img_display = aug_img.clone()
            for t, m, s in zip(aug_img_display, MEAN, STD):
                t.mul_(s).add_(m)
            aug_img_display = torch.clamp(aug_img_display, 0, 1)
            
            # convert to PIL image
            aug_img_np = aug_img_display.permute(1, 2, 0).numpy()
            aug_img_np = (aug_img_np * 255).astype(np.uint8)
            aug_img_pil = Image.fromarray(aug_img_np)
            
            # save image
            filename = f"{class_name}_aug_{i+1}.png"
            filepath = graphics_dir / filename
            aug_img_pil.save(filepath)
            
            print(f"  saved: {filename}")
    
    print("done saving augmented samples")


def create_dataloaders(
    splits: Dict,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS
) -> Dict[str, DataLoader]:
    
    ## creates data-loaders ##
    
    datasets = {
        'train': BrainTumorDataset(
            splits['train'][0],
            splits['train'][1],
            transform=get_train_transforms()
        ),
        'val': BrainTumorDataset(
            splits['val'][0],
            splits['val'][1],
            transform=get_val_test_transforms()
        ),
        'test': BrainTumorDataset(
            splits['test'][0],
            splits['test'][1],
            transform=get_val_test_transforms()
        )
    }
    
    # dataloaders
    dataloaders = {
        'train': DataLoader(
            datasets['train'],
            batch_size=batch_size,
            shuffle=True,  
            num_workers=num_workers,
            pin_memory=PIN_MEMORY,
            persistent_workers=PERSISTENT_WORKERS if num_workers > 0 else False,
            drop_last=True  
        ),
        'val': DataLoader(
            datasets['val'],
            batch_size=batch_size,
            shuffle=False,  
            num_workers=num_workers,
            pin_memory=PIN_MEMORY,
            persistent_workers=PERSISTENT_WORKERS if num_workers > 0 else False
        ),
        'test': DataLoader(
            datasets['test'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=PIN_MEMORY,
            persistent_workers=PERSISTENT_WORKERS if num_workers > 0 else False
        )
    }
    
    print("\n dataloaders created")
    print(f"Training batches: {len(dataloaders['train'])}")
    print(f"Validation batches: {len(dataloaders['val'])}")
    print(f"Test batches: {len(dataloaders['test'])}")
    
    return dataloaders, datasets


def process_and_prepare_data(
    raw_data_dir: Path = RAW_DIR,
    output_dir: Path = PROCESSED_DIR,
    val_size: float = 0.15,
    test_size: float = 0.15
) -> Tuple[Dict[str, DataLoader], Dict[str, Dataset]]:
    
    ## process data ##

    # collect all data
    print("\ncollecting image paths...")
    image_paths, labels = collect_data_paths(raw_data_dir)
    print(f"found {len(image_paths)} images across {NUM_CLASSES} classes")
    
    if len(image_paths) == 0:
        raise ValueError(f"no images found in {raw_data_dir}!")
    
    # create splits
    print("\ncreating train/val/test splits...")
    splits = create_splits(image_paths, labels, val_size, test_size)
    
    
    # create dataloaders
    print("\ncreating dataLoaders...")
    dataloaders, datasets = create_dataloaders(splits)
    
    print("done")
    print(f"train samples: {len(splits['train'][0])}")
    print(f"val samples: {len(splits['val'][0])}")
    print(f"test samples: {len(splits['test'][0])}")
    
    return dataloaders, datasets


if __name__ == "__main__":
    
    ## run ts ##
    
    # process data and create dataloaders
    dataloaders, datasets = process_and_prepare_data()
    
    # save 3 augmented samples from each class
    save_augmented_samples(datasets['train'])
    
    # vtest batch-loading
    print("\nloading sample batch...")
    
    train_loader = dataloaders['train']
    images, labels = next(iter(train_loader))
    
    print("\ndata pipeline ready for training.")