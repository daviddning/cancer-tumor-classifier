## configuration file for brain tumor classification ##

import os
from pathlib import Path
import torch
import numpy as np
import random

## data paths ##
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" 
RAW_DIR = DATA_DIR / 'raw' 
PROCESSED_DIR = DATA_DIR / 'processed'
TRAIN_DIR = PROCESSED_DIR / 'train'
TEST_DIR = PROCESSED_DIR / 'test'
GRAPHICS_DIR = BASE_DIR / "graphics"
GRADCAM_DIR = GRAPHICS_DIR / 'gradcam'
CHECKPOINT_DIR = BASE_DIR / 'checkpoints'

## model classes ##
CLASSES = ['glioma', 'meningioma', 'pituitary']
NUM_CLASSES = len(CLASSES)

## data splitting ##
VAL_SIZE = 0.15
TEST_SIZE = 0.15
SPLIT_RANDOM_STATE = 88

## augmentation settings ##
AUG_SAMPLES_PER_CLASS = 3
ROTATION_DEGREES = 20
GAUSSIAN_BLUR_PROB = 0.5
VERTICAL_FLIP_PROB = 0.3
HORIZONTAL_FLIP_PROB = 0.5

## training hyperparameters ##
BATCH_SIZE = 32 
NUM_WORKERS = 4 
PIN_MEMORY = True  
PERSISTENT_WORKERS = False
PREFETCH_FACTOR = 2
DROP_LAST_TRAIN = False
GRADIENT_CLIP = 1.0  
LABEL_SMOOTHING = 0.1 
WEIGHT_DECAY = 0.01
DROPOUT_RATE = 0.4
USE_MIXED_PRECISION = True

## training phases ##
STAGE1_EPOCHS = 5
STAGE2_EPOCHS = 15
STAGE3_EPOCHS = 25

## learning rate ##
LR_STAGE1 = 1e-3
LR_STAGE2 = 1e-5    
LR_STAGE3 = 1e-6

## stage 2 lr factors ##
STAGE2_BACKBONE_LR_FACTOR = 0.1
LR_STAGE2_BACKBONE = LR_STAGE2 * STAGE2_BACKBONE_LR_FACTOR
LR_STAGE2_HEAD = LR_STAGE2

## stage 3 lr factors ##
STAGE3_LAYER1_LR_FACTOR = 0.01
STAGE3_LAYER2_LR_FACTOR = 0.02
STAGE3_LAYER3_LR_FACTOR = 0.05
STAGE3_LAYER4_LR_FACTOR = 0.1
LR_STAGE3_LAYER1 = LR_STAGE3 * STAGE3_LAYER1_LR_FACTOR
LR_STAGE3_LAYER2 = LR_STAGE3 * STAGE3_LAYER2_LR_FACTOR
LR_STAGE3_LAYER3 = LR_STAGE3 * STAGE3_LAYER3_LR_FACTOR
LR_STAGE3_LAYER4 = LR_STAGE3 * STAGE3_LAYER4_LR_FACTOR
LR_STAGE3_HEAD = LR_STAGE3

## early stopping ##
PATIENCE_STAGE2 = 7
PATIENCE_STAGE3 = 10
MIN_DELTA_STAGE2 = 0.002
MIN_DELTA_STAGE3 = 0.0007

## scheduler parameters ##
STAGE1_T0 = 5
STAGE1_T_MULT = 1
STAGE2_LR_FACTOR = 0.5
STAGE2_SCHEDULER_PATIENCE = 3
STAGE3_LR_FACTOR = 0.3
STAGE3_SCHEDULER_PATIENCE = 5

## image settings ##
IMAGE_SIZE = 384  
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

## grad-cam configuration ##
GRADCAM_CONFIG = {
    'target_layers': ['layer3.5', 'layer4.0', 'layer4.2'],  
    'colormap': 'jet',            
    'alpha': 0.5,               
    'use_cuda': True,            
    'figure_size': (20, 5),
    'save_format': 'png',
    'dpi': 300,
    'samples_per_class': 2,        
    'include_incorrect': True,     
    'checkpoints': {
        'stage1_start': True,
        'stage1_end': True, 
        'stage2_start': True,
        'stage2_end': True,
        'stage3_start': True,
        'stage3_interval': 5,      
    }
}

## device ##
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRETRAINED_WEIGHTS_PATH = None

## seed for reproducibility ##
RANDOM_SEED = 88
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

## create directories ##
for directory in [DATA_DIR, RAW_DIR, PROCESSED_DIR, GRAPHICS_DIR, GRADCAM_DIR, CHECKPOINT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)