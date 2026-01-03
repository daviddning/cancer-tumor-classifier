import pandas as pd
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json
from tqdm import tqdm
import config
import warnings
warnings.filterwarnings('ignore')

def create_patient_wise_split(metadata_df, config):
    """
    split data by patients to prevent data leakage.
    returns train, val, test patient IDs and saves to disk.
    """
    print("\ncreating patient-wise data split")
    
    # get unique patients
    patients = metadata_df['volume'].unique()
    n_patients = len(patients)
    
    print(f"\nTotal patients: {n_patients}")
    print(f"Train ratio: {config.TRAIN_RATIO:.1%}")
    print(f"Val ratio: {config.VAL_RATIO:.1%}")
    print(f"Test ratio: {config.TEST_RATIO:.1%}")
    
    # shuffle patients
    np.random.seed(config.RANDOM_SEED)
    np.random.shuffle(patients)
    
    # calculate split indices
    n_train = int(n_patients * config.TRAIN_RATIO)
    n_val = int(n_patients * config.VAL_RATIO)
    
    # split patients
    train_patients = patients[:n_train]
    val_patients = patients[n_train:n_train + n_val]
    test_patients = patients[n_train + n_val:]
    
    print(f"\nTrain patients: {len(train_patients)} ({len(train_patients)/n_patients:.1%})")
    print(f"Val patients: {len(val_patients)} ({len(val_patients)/n_patients:.1%})")
    print(f"Test patients: {len(test_patients)} ({len(test_patients)/n_patients:.1%})")
    
    # filter metadata by patient split
    train_df = metadata_df[metadata_df['volume'].isin(train_patients)].reset_index(drop=True)
    val_df = metadata_df[metadata_df['volume'].isin(val_patients)].reset_index(drop=True)
    test_df = metadata_df[metadata_df['volume'].isin(test_patients)].reset_index(drop=True)
    
    # print slice statistics
    print(f"\nTrain slices: {len(train_df):,} (Tumor: {train_df['target'].sum():,})")
    print(f"Val slices: {len(val_df):,} (Tumor: {val_df['target'].sum():,})")
    print(f"Test slices: {len(test_df):,} (Tumor: {test_df['target'].sum():,})")
    
    # calculate tumor percentages
    train_tumor_pct = train_df['target'].mean() * 100
    val_tumor_pct = val_df['target'].mean() * 100
    test_tumor_pct = test_df['target'].mean() * 100
    
    print(f"\nTumor distribution:")
    print(f"  Train: {train_tumor_pct:.1f}%")
    print(f"  Val: {val_tumor_pct:.1f}%")
    print(f"  Test: {test_tumor_pct:.1f}%")
    
    # save split information
    config.SPLIT_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    
    split_info = {
        'train_patients': train_patients.tolist(),
        'val_patients': val_patients.tolist(),
        'test_patients': test_patients.tolist(),
        'random_seed': config.RANDOM_SEED,
        'train_tumor_percentage': float(train_tumor_pct),
        'val_tumor_percentage': float(val_tumor_pct),
        'test_tumor_percentage': float(test_tumor_pct)
    }
    
    with open(config.SPLIT_SAVE_PATH / 'patient_split.json', 'w') as f:
        json.dump(split_info, f, indent=2)
    
    # save metadata splits
    train_df.to_csv(config.SPLIT_SAVE_PATH / 'train_metadata.csv', index=False)
    val_df.to_csv(config.SPLIT_SAVE_PATH / 'val_metadata.csv', index=False)
    test_df.to_csv(config.SPLIT_SAVE_PATH / 'test_metadata.csv', index=False)
    
    print(f"\nSplit information saved to: {config.SPLIT_SAVE_PATH}")
    
    return train_df, val_df, test_df

# file path utilities
def get_file_paths_from_metadata(metadata_df, data_path):
    """
    convert metadata entries to actual .h5 file paths.
    handles different possible path structures.
    """
    file_paths = []
    missing_files = []
        
    for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Finding files"):
        # try to get path from metadata
        if 'slice_path' in row and pd.notna(row['slice_path']):
            slice_name = Path(row['slice_path']).name
        else:
            # construct filename from volume and slice number
            volume_id = row['volume']
            slice_num = row['slice']
            slice_name = f"volume_{volume_id}_slice_{slice_num}.h5"
        
        # search for the file
        h5_files = list(data_path.rglob(slice_name))
        
        if h5_files:
            file_paths.append(str(h5_files[0]))
        else:
            missing_files.append(slice_name)
            file_paths.append(None)
    
    # report results
    valid_files = [f for f in file_paths if f is not None]
    print(f"\nfound {len(valid_files):,} / {len(metadata_df):,} files")
    
    if missing_files:
        print(f"   warning: {len(missing_files)} files not found")
        print(f"   first few missing: {missing_files[:5]}")
    
    return file_paths


# dataset
class BraTSDataset(Dataset):
    """
    PyTorch Dataset for BraTS 2020 brain tumor segmentation.
    loads 4-channel MRI images and binary tumor masks from .h5 files.
    """
    
    def __init__(self, file_paths, metadata_df, transform=None, normalization="z_score"):
        """
        Args:
            file_paths: List of .h5 file paths
            metadata_df: dataframe with metadata for these files
            transform: transform pipeline
            normalization: Normalization strategy 
        """
        # filter out missing files
        valid_indices = [i for i, path in enumerate(file_paths) if path is not None]
        
        self.file_paths = [file_paths[i] for i in valid_indices]
        self.metadata = metadata_df.iloc[valid_indices].reset_index(drop=True)
        self.transform = transform
        self.normalization = normalization
        
        print(f"dataset initialized with {len(self.file_paths)} valid samples")
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        """load and return a single MRI slice with its mask"""
        try:
            # load .h5 file
            with h5py.File(self.file_paths[idx], 'r') as f:
                image = f['image'][:]  # shape: (240, 240, 4)
                mask = f['mask'][:]    # shape: (240, 240)
            
            # normalize image
            image = self.normalize_image(image)
            
            # convert mask to binary 
            mask = (mask > 0).astype(np.float32)
            
            # apply augmentations 
            if self.transform:
                transformed = self.transform(image=image, mask=mask)
                image = transformed['image']
                mask = transformed['mask']
            
            # convert to PyTorch tensors
            # image: (4, 240, 240), Mask: (1, 240, 240)
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
            mask = torch.from_numpy(mask).unsqueeze(0).float()
            
            return image, mask
        
        except Exception as e:
            print(f"error loading file {self.file_paths[idx]}: {e}")
            # return zeros if file fails to load
            return torch.zeros(4, 240, 240), torch.zeros(1, 240, 240)
    
    def normalize_image(self, image):
        """
        normalize each MRI modality independently.
        
        Args:
            image: numpy array of shape (H, W, 4)
        Returns:
            normalized image of same shape
        """
        normalized = np.zeros_like(image, dtype=np.float32)
        
        if self.normalization == "z_score":
            # z-score normalization per channel
            for i in range(4):
                channel = image[:, :, i].astype(np.float32)
                mean = channel.mean()
                std = channel.std()
                if std > 0:
                    normalized[:, :, i] = (channel - mean) / std
                else:
                    normalized[:, :, i] = channel - mean
        
        elif self.normalization == "min_max":
            # min-max normalization to [0, 1] per channel
            for i in range(4):
                channel = image[:, :, i].astype(np.float32)
                min_val = channel.min()
                max_val = channel.max()
                if max_val > min_val:
                    normalized[:, :, i] = (channel - min_val) / (max_val - min_val)
                else:
                    normalized[:, :, i] = 0
        
        elif self.normalization == "per_patient":
            # normalize using patient-level statistics
            # (this would require pre-computed stats, simplified here)
            for i in range(4):
                channel = image[:, :, i].astype(np.float32)
                normalized[:, :, i] = (channel - channel.mean()) / (channel.std() + 1e-8)
        
        return normalized


# augmentation helpers
def get_training_augmentation(config):
    """
    returns Albumentations pipeline for training data.
    includes geometric and intensity augmentations suitable for medical imaging.
    """
    return A.Compose([
        # geometric augmentations
        A.HorizontalFlip(p=config.HORIZONTAL_FLIP_PROB),
        A.Rotate(limit=config.ROTATION_LIMIT, p=0.5, border_mode=0),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=config.ROTATION_LIMIT,
            p=0.3,
            border_mode=0
        ),
        
        # intensity augmentations 
        A.RandomBrightnessContrast(
            brightness_limit=0.1,
            contrast_limit=0.1,
            p=config.BRIGHTNESS_CONTRAST_PROB
        ),
        A.GaussNoise(
            var_limit=(0.0, 0.01),
            p=config.GAUSSIAN_NOISE_PROB
        ),
        
        # elastic deformation 
        A.ElasticTransform(
            alpha=50,
            sigma=5,
            alpha_affine=5,
            p=0.2,
            border_mode=0
        ),
    ])

def get_validation_augmentation():
    """
    returns empty pipeline for validation/test data.
    no augmentation
    """
    return None


# dataloaders
def create_dataloaders(config, use_augmentation=True):
    """
    function to create train, validation, and test dataLoaders.
    
    Args:
        config: Configuration object
        use_augmentation: Whether to apply augmentation to training data
    
    Returns:
        train_loader, val_loader, test_loader
    """
    print("\ncreating dataloaders")
    
    # load metadata
    print(f"\nloading metadata from: {config.METADATA_PATH}")
    metadata_df = pd.read_csv(config.METADATA_PATH)
    print(f"loaded {len(metadata_df):,} total slices")
    
    # Cceate patient-wise split
    train_df, val_df, test_df = create_patient_wise_split(metadata_df, config)
    
    # get file paths for each split
    print("\nTRAIN SET")
    train_files = get_file_paths_from_metadata(train_df, config.DATA_PATH)
    
    print("\nVALIDATION SET")
    val_files = get_file_paths_from_metadata(val_df, config.DATA_PATH)
    
    print("\nTEST SET")
    test_files = get_file_paths_from_metadata(test_df, config.DATA_PATH)
    
    # create augmentation pipelines
    if use_augmentation:
        train_transform = get_training_augmentation(config)
        print("\ntraining augmentation enabled")
    else:
        train_transform = None
        print("\ntraining augmentation disabled")
    
    val_transform = get_validation_augmentation()
    test_transform = get_validation_augmentation()
    
    # create datasets
    print("\ncreating datasets...")
    train_dataset = BraTSDataset(
        train_files, 
        train_df, 
        transform=train_transform,
        normalization=config.NORMALIZATION
    )
    val_dataset = BraTSDataset(
        val_files, 
        val_df, 
        transform=val_transform,
        normalization=config.NORMALIZATION
    )
    test_dataset = BraTSDataset(
        test_files, 
        test_df, 
        transform=test_transform,
        normalization=config.NORMALIZATION
    )
    
    # create dataloaders
    print("\ncreating DataLoaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    # print summary
    print("DATALOADER SUMMARY")
    print(f"\nTrain DataLoader:")
    print(f"  - Batches: {len(train_loader)}")
    print(f"  - Samples: {len(train_dataset)}")
    print(f"  - Batch size: {config.BATCH_SIZE}")
    print(f"  - Shuffle: True")
    
    print(f"\nValidation DataLoader:")
    print(f"  - Batches: {len(val_loader)}")
    print(f"  - Samples: {len(val_dataset)}")
    print(f"  - Batch size: {config.BATCH_SIZE}")
    print(f"  - Shuffle: False")
    
    print(f"\nTest DataLoader:")
    print(f"  - Batches: {len(test_loader)}")
    print(f"  - Samples: {len(test_dataset)}")
    print(f"  - Batch size: {config.BATCH_SIZE}")
    print(f"  - Shuffle: False")
    
    return train_loader, val_loader, test_loader

# verification  
def verify_dataloader(dataloader, name="DataLoader"):
    """
    verify that the dataloader works correctly by loading a batch.
    """
    print(f"\nVERIFYING {name}")
    try:
        # get first batch
        images, masks = next(iter(dataloader))
        
        print(f"\nSuccessfully loaded batch")
        print(f"  - Image shape: {images.shape}")  # Expected: (B, 4, 240, 240)
        print(f"  - Mask shape: {masks.shape}")    # Expected: (B, 1, 240, 240)
        print(f"  - Image dtype: {images.dtype}")
        print(f"  - Mask dtype: {masks.dtype}")
        print(f"  - Image range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"  - Mask range: [{masks.min():.3f}, {masks.max():.3f}]")
        print(f"  - Tumor pixels in batch: {masks.sum().item():.0f}")
        
        # check for NaN or Inf
        if torch.isnan(images).any():
            print("  ⚠ WARNING: NaN values detected in images!")
        if torch.isinf(images).any():
            print("  ⚠ WARNING: Inf values detected in images!")
        
        return True
    
    except Exception as e:
        print(f"\nError loading batch: {e}")
        return False


# main
if __name__ == "__main__":

    # create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(config, use_augmentation=True)
    
    # verify each dataloader
    verify_dataloader(train_loader, "TRAIN DATALOADER")
    verify_dataloader(val_loader, "VALIDATION DATALOADER")
    verify_dataloader(test_loader, "TEST DATALOADER")
    
    print("\nDATA PREPROCESSING COMPLETE")