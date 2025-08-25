import os
import random
from pathlib import Path
from typing import List, Tuple, Dict
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
from config import (
    TRAIN_DIR, TEST_DIR, VAL_SPLIT, RANDOM_STATE,
    CLASSES, IMAGE_SIZE, MEAN, SD, AUTOAUGMENT
)

# directories
DATA_DIR = Path(__file__).parent.parent / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'
PROCESSED_TRAIN_DIR = PROCESSED_DIR / 'training'
PROCESSED_VAL_DIR = PROCESSED_DIR / 'validation'
PROCESSED_TEST_DIR = PROCESSED_DIR / 'testing'

# random seeds for reproducibility
torch.manual_seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

def get_train_transform() -> transforms.Compose:
    """create transforms for training data with augmentation."""
    augs = [
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(15),
    ]
    
    if AUTOAUGMENT:
        augs.append(AutoAugment(AutoAugmentPolicy.IMAGENET))
    
    augs.extend([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, SD)
    ])
    
    return transforms.Compose(augs)

def get_val_test_transform() -> transforms.Compose:
    """transforms for validation/test data (no augmentation)."""
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, SD)
    ])

# Initialize transforms
train_transform = get_train_transform()
val_test_transform = get_val_test_transform()

def get_class_from_foldername(filepath: Path) -> str:
    """extract class name from directory path."""
    foldername = filepath.parent.name.lower()
    
    # Map folder names to class names
    if 'glioma' in foldername:
        return 'glioma'
    if 'meningioma' in foldername:
        return 'meningioma'
    if 'pituitary' in foldername:
        return 'pituitary'
    if 'no' in foldername and 'tumor' in foldername:
        return 'no_tumor'
    
    # fallback to exact match if needed
    for cls in CLASSES:
        if cls.lower() in foldername:
            return cls
    
    raise ValueError(f"unknown class for folder: {foldername}")

def collect_files_by_class(data_dir: Path) -> Dict[str, List[Path]]:
    """gather all image files, organized by class."""
    if not data_dir.exists():
        raise FileNotFoundError(f"dataset directory not found: {data_dir}")
    
    # image formats
    img_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    class_files = {cls: [] for cls in CLASSES}
    
    for class_dir in data_dir.iterdir():
        if not class_dir.is_dir():
            continue
            
        # find all image files in this directory
        images = []
        for ext in img_exts:
            images.extend(class_dir.glob(f'*{ext}'))
            
        if not images:
            print(f"no images found in {class_dir}")
            continue
            
        try:
            # get class name from directory 
            class_name = get_class_from_foldername(images[0])
            if class_name not in CLASSES:
                print(f"skipping unknown class: {class_dir.name}")
                continue
                
            class_files[class_name] = images
            print(f"found {len(images)} {class_name} images")
            
        except Exception as e:
            print(f"error in {class_dir}: {str(e)}")
    
    return class_files

def split_data(class_files: Dict[str, List[Path]]) -> tuple[list, list]:
    """split data into training and validation sets."""
    train_files, val_files = [], []
    
    for class_name, files in class_files.items():
        if not files:
            continue
            
        # split this class's files
        train, val = train_test_split(
            files,
            test_size=VAL_SPLIT,
            random_state=RANDOM_STATE,
            shuffle=True
        )
        
        # add (file, class) pairs
        train_files.extend((f, class_name) for f in train)
        val_files.extend((f, class_name) for f in val)
    
    # shuffle to mix up the classes
    random.shuffle(train_files)
    random.shuffle(val_files)
    
    print(f"Training: {len(train_files)} images")
    print(f"Validation: {len(val_files)} images")
    
    return train_files, val_files

def process_and_save(file_class_list: list[tuple[Path, str]], 
                    dest_dir: Path, 
                    transform: transforms.Compose,
                    split_name: str = '') -> None:
    """process images and save them as tensors."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    processed = 0
    
    for i, (img_path, class_name) in enumerate(file_class_list, 1):
        try:
            # load and transform image
            img = Image.open(img_path).convert('RGB')
            tensor = transform(img)
            
            # save to class directory
            save_dir = dest_dir / class_name
            save_dir.mkdir(exist_ok=True)
            torch.save(tensor, save_dir / f"{img_path.stem}.pt")
            
            processed += 1
            if i % 100 == 0 or i == len(file_class_list):
                print(f"{split_name}: Processed {i}/{len(file_class_list)}")
                
        except Exception as e:
            print(f"Error with {img_path.name}: {str(e)}")
    
    print(f"finished. Processed {processed}/{len(file_class_list)} images for {split_name}")

def load_test_files() -> list[tuple[Path, str]]:
    """load test images and their corresponding classes."""
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Test directory not found: {TEST_DIR}")
    
    test_files = []
    img_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    
    for class_dir in TEST_DIR.iterdir():
        if not class_dir.is_dir():
            continue
            
        # find all test images
        images = []
        for ext in img_exts:
            images.extend(class_dir.glob(f'*{ext}'))
            
        if not images:
            print(f"no test images in {class_dir.name}")
            continue
            
        try:
            class_name = get_class_from_foldername(images[0])
            if class_name not in CLASSES:
                print(f"skipping unknown test class: {class_dir.name}")
                continue
                
            test_files.extend((img, class_name) for img in images)
            print(f"found {len(images)} {class_name} test images")
            
        except Exception as e:
            print(f"error loading test images from {class_dir}: {str(e)}")
    
    return test_files

def main():
    print("starting data processing.")
    
    try:
        # output directories
        for d in [PROCESSED_TRAIN_DIR, PROCESSED_VAL_DIR, PROCESSED_TEST_DIR]:
            d.mkdir(parents=True, exist_ok=True)
        
        # process training and validation data
        print("\nloading training data.")
        train_data = collect_files_by_class(TRAIN_DIR)
        train_files, val_files = split_data(train_data)
        
        # process test data
        print("\nloading test data.")
        test_files = load_test_files()
        
        # process and save all datasets
        print("\nprocessing datasets.")
        process_and_save(train_files, PROCESSED_TRAIN_DIR, train_transform, "Training")
        process_and_save(val_files, PROCESSED_VAL_DIR, val_test_transform, "Validation")
        process_and_save(test_files, PROCESSED_TEST_DIR, val_test_transform, "Testing")
        
        # summary
        print("data processing complete")
        print(f"training:   {len(train_files):5d} images")
        print(f"validation: {len(val_files):5d} images")
        print(f"test:       {len(test_files):5d} images")
        
    except Exception as e:
        print(f"\nerror: {str(e)}")
        raise
if __name__ == "__main__":
    main()
