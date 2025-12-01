import os
from pathlib import Path
import torch
import numpy as np
import random

# data paths
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent
DATA_DIR = BASE_DIR / "data" 
RAW_DIR = DATA_DIR / 'raw' 
PROCESSED_DIR = DATA_DIR / 'processed'
TRAIN_DIR = PROCESSED_DIR / 'train'
TEST_DIR = PROCESSED_DIR / 'test'
GRAPHICS_DIR = BASE_DIR / "graphics"
GRADCAM_DIR = BASE_DIR / 'graphics' / 'gradcam'
CHECKPOINT_DIR = BASE_DIR / 'checkpoints'

# model classes
CLASSES = ['glioma', 'meningioma', 'pituitary']
NUM_CLASSES = len(CLASSES)

# training hyperparameters
BATCH_SIZE = 32 
NUM_WORKERS = 4 
PIN_MEMORY = True  
PERSISTENT_WORKERS = True 
GRADIENT_CLIP = 1.0  
LABEL_SMOOTHING = 0.1 

# training phases
STAGE1_EPOCHS = 5
STAGE2_EPOCHS = 15
STAGE3_EPOCHS = 25

# learning rate
LR_STAGE1 = 1e-3
LR_STAGE2 = 1e-5    
LR_STAGE3 = 1e-6
WEIGHT_DECAY = 0.01

# early stoppage
PATIENCE_STAGE2 = 7
PATIENCE_STAGE3 = 10
MIN_DELTA_STAGE2 = 0.002
MIN_DELTA_STAGE3 = 0.0007

DROPOUT_RATE = 0.4

# normalization
IMAGE_SIZE = 384  
MEAN = [0.482, 0.458, 0.408]
STD = [0.231, 0.226, 0.225]

# Grad-CAM 
GRADCAM_CONFIG = {
    'target_layers': ['layer4.2'],  
    'colormap': 'jet',            
    'alpha': 0.5,               
    'use_cuda': True,            
    
    # visualization settings
    'figure_size': (15, 5),
    'save_format': 'png',
    'dpi': 300,
    
    # samples
    'samples_per_class': 2,        
    'include_incorrect': True,     
    
    # checkpoint schedule
    'checkpoints': {
        'stage1_start': True,
        'stage1_end': True, 
        'stage2_start': True,
        'stage2_end': True,
        'stage3_start': True,
        'stage3_interval': 5,      
    }
}

# metrics
METRICS = ['accuracy', 'f1_score', 'precision', 'recall']

DEVICE = torch.device("cuda")
MODEL_SAVE_PATH = BASE_DIR / "models" / "brain_tumor_model.pth"

# seed for reproducibility
torch.manual_seed(88)
np.random.seed(88)
random.seed(88)
