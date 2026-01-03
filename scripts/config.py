from pathlib import Path
import torch

DATA_PATH = Path.cwd() / "data"
METADATA_PATH = DATA_PATH / "csv" / "meta_data.csv"
SPLIT_SAVE_PATH = DATA_PATH / "splits"
    
# data split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 88
    
# model parameters
INPUT_CHANNELS = 4  # FLAIR, T1, T1ce, T2
IMAGE_SIZE = 240
NUM_CLASSES = 1  
    
# training parameters
BATCH_SIZE = 16
NUM_WORKERS = 4
    
# augmentation parameters
HORIZONTAL_FLIP_PROB = 0.5
ROTATION_LIMIT = 15
BRIGHTNESS_CONTRAST_PROB = 0.3
GAUSSIAN_NOISE_PROB = 0.2
    
# normalization strategy
NORMALIZATION = "z_score"

BASE_FILTERS = 64           
DROPOUT_RATE = 0.2          
USE_BATCH_NORM = True      
ACTIVATION = 'relu'         

# attention hyperparameters
ATTENTION_REDUCTION = 2     

# output hyperparameters
OUTPUT_ACTIVATION = 'sigmoid'  

# grad-CAM hyperparameters
GRADCAM_LAYER = 'bottleneck'  
GRADCAM_SAVE_PATH = Path.cwd() / "graphics" / "gradcam"
GRADCAM_SAVE_PATH.mkdir(parents=True, exist_ok=True)