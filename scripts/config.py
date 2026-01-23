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
INPUT_CHANNELS = 4  
IMAGE_SIZE = 240
NUM_CLASSES = 1  
    
# training parameters
BATCH_SIZE = 4
NUM_WORKERS = 6
    
# augmentation parameters
HORIZONTAL_FLIP_PROB = 0.5
ROTATION_LIMIT = 15
BRIGHTNESS_CONTRAST_PROB = 0.3
GAUSSIAN_NOISE_PROB = 0.2
    
# normalization strategy
NORMALIZATION = "z_score"

BASE_FILTERS = 32          
DROPOUT_RATE = 0.2          
USE_BATCH_NORM = True      
ACTIVATION = 'relu'         

# attention hyperparameters
ATTENTION_REDUCTION = 2     

# output hyperparameters
OUTPUT_ACTIVATION = 'sigmoid'  

# loss hyperparameters
# dice loss
DICE_SMOOTH = 1.0

# focal loss
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0
FOCAL_REDUCTION = 'mean'

# bce loss
POS_WEIGHT = 1.3  

# combined loss
DICE_WEIGHT = 0.5
BCE_WEIGHT = 0.3
FOCAL_WEIGHT = 0.2