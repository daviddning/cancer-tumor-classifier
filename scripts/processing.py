import os
import random
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from sklearn.model_selection import train_test_split
import shutil

# Set paths
RAW_DATA_DIR = r"C:\Users\SCHOOL\OneDrive - UBC\Documents\brain-tumor-classifier\data\raw"
PROCESSED_TRAIN_DIR = r"C:\Users\SCHOOL\OneDrive - UBC\Documents\brain-tumor-classifier\data\processed\training"
PROCESSED_VAL_DIR = r"C:\Users\SCHOOL\OneDrive - UBC\Documents\brain-tumor-classifier\data\processed\validation"
PROCESSED_TEST_DIR = r"C:\Users\SCHOOL\OneDrive - UBC\Documents\brain-tumor-classifier\data\processed\kaggle-testing"

# Train/val/test split ratio
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# AutoAugment policy (ImageNet policy works well for MRIs too)
augmentation_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
    transforms.ToTensor()
])

# Standard preprocessing transform
standard_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(num_output_channels=3),  # Convert to RGB
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
])

def copy_files(files, src_dir, dst_dir):
    """Copy files from src to dst"""
    for f in files:
        src_path = os.path.join(src_dir, f)
        dst_path = os.path.join(dst_dir, f)

        try:
            img = Image.open(src_path).convert("L")  # Grayscale MRI
            img = img.resize((256, 256))  # Resize before saving
            img.save(dst_path)  # Save pre-resized image
        except Exception as e:
            pass  # Silent fail on invalid file

def augment_class(class_name, src_dir, dst_dir, num_augmented_samples=100):
    """Generate augmented samples for a class"""
    class_src = os.path.join(src_dir, class_name)
    class_dst = os.path.join(dst_dir, class_name)

    files = [f for f in os.listdir(class_src) if os.path.isfile(os.path.join(class_src, f))]

    for i in range(num_augmented_samples):
        try:
            img_path = os.path.join(class_src, random.choice(files))
            img = Image.open(img_path).convert("RGB")  # Convert to RGB for augmentation
            augmented_img = augmentation_transform(img)
            augmented_img = transforms.ToPILImage()(augmented_img)

            save_path = os.path.join(class_dst, f"{class_name}_aug_{i}.jpg")
            augmented_img.save(save_path)
        except Exception as e:
            continue  # Silent fail on any augmentation error

def process_and_save():
    class_names = sorted(os.listdir(RAW_DATA_DIR))

    for cls in class_names:
        src_cls_dir = os.path.join(RAW_DATA_DIR, cls)
        files = [f for f in os.listdir(src_cls_dir) if os.path.isfile(os.path.join(src_cls_dir, f))]

        # Stratified train/val/test split
        train_files, test_files = train_test_split(files, test_size=TEST_RATIO, random_state=42)
        train_files, val_files = train_test_split(train_files, test_size=VAL_RATIO / (TRAIN_RATIO + VAL_RATIO), random_state=42)

        # Copy files
        [copy_files(train_files, src_cls_dir, os.path.join(PROCESSED_TRAIN_DIR, cls))]
        [copy_files(val_files, src_cls_dir, os.path.join(PROCESSED_VAL_DIR, cls))]
        [copy_files(test_files, src_cls_dir, os.path.join(PROCESSED_TEST_DIR, cls))]

        # Apply augmentation to training set
        augment_class(cls, src_cls_dir, PROCESSED_TRAIN_DIR, num_augmented_samples=200)

# Run processing silently
process_and_save()