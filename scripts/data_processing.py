import os
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).parent.parent.resolve()

RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_TRAIN_DIR = BASE_DIR / "data" / "processed" / "training"
PROCESSED_VAL_DIR = BASE_DIR / "data" / "processed" / "validation"
PROCESSED_TEST_DIR = BASE_DIR / "data" / "processed" / "testing-1"

CLASSES = ['glioma', 'meningioma', 'pituitary']

TRAIN_RATIO = 0.75
VAL_RATIO = 0.10
TEST_RATIO = 0.15

### data transformations ###

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((384, 384)),
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((384, 384)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def get_class_from_foldername(filepath: Path) -> str:
    foldername = filepath.parent.name.lower()
    if 'glioma' in foldername:
        return 'glioma'
    elif 'menin' in foldername:
        return 'meningioma'
    elif 'pituitary' in foldername:
        return 'pituitary'
    else:
        raise ValueError(f"unknown class for folder: {foldername}")


def collect_files_by_class(raw_data_dir: Path) -> dict:
    class_files = {cls: [] for cls in CLASSES}
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}

    for filepath in raw_data_dir.rglob("*"):
        if filepath.is_file() and filepath.suffix.lower() in valid_extensions:
            try:
                class_name = get_class_from_foldername(filepath)
                class_files[class_name].append(filepath)
            except ValueError as e:
                print(f"skipping file {filepath.name}: {e}")
    return class_files


def split_data(class_files: dict):
    train_files, val_files, test_files = [], [], []

    for class_name, files in class_files.items():
        class_train, class_test = train_test_split(
            files, test_size=TEST_RATIO, random_state=55)
        class_train, class_val = train_test_split(
            class_train, test_size=VAL_RATIO / (TRAIN_RATIO + VAL_RATIO), random_state=55)

        train_files.extend([(f, class_name) for f in class_train])
        val_files.extend([(f, class_name) for f in class_val])
        test_files.extend([(f, class_name) for f in class_test])

    return train_files, val_files, test_files


def process_and_save(file_class_list, dest_dir: Path, transform):
    for fpath, class_name in file_class_list:
        try:
            img = Image.open(fpath).convert('RGB')
            img_np = np.array(img)  

            tensor = transform(img_np)

            class_dir = dest_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)  

            save_name = f"{fpath.stem}.pt"
            save_path = class_dir / save_name

            torch.save(tensor, save_path)


        except Exception as e:
            print(f"failed to process {fpath}: {e}")


def main():
    print(f"collecting files from {RAW_DATA_DIR}...")
    class_files = collect_files_by_class(RAW_DATA_DIR)

    total_files = sum(len(files) for files in class_files.values())
    print(f"found {total_files} images total.")
    for cls, files in class_files.items():
        print(f"{cls}: {len(files)} images")

    train_files, val_files, test_files = split_data(class_files)

    print("\nfinal counts:")
    print(f"train: {len(train_files)}")
    print(f"validation: {len(val_files)}")
    print(f"test: {len(test_files)}")

    print("\nprocessing training set...")
    process_and_save(train_files, PROCESSED_TRAIN_DIR, train_transform)

    print("processing validation set...")
    process_and_save(val_files, PROCESSED_VAL_DIR, val_test_transform)

    print("processing test set...")
    process_and_save(test_files, PROCESSED_TEST_DIR, val_test_transform)

    print("\ndone processing and saving all images.")
    print(f"training images saved to: {PROCESSED_TRAIN_DIR}")
    print(f"validation images saved to: {PROCESSED_VAL_DIR}")
    print(f"test images saved to: {PROCESSED_TEST_DIR}")


if __name__ == "__main__":
    main()
