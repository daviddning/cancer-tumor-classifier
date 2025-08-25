import os
from pathlib import Path
import torch
import numpy as np
import random

# config
BASE_DIR = Path(__file__).parent.parent.resolve()

# data paths
DATA_DIR = Path(r"C:\Users\SCHOOL\.cache\kagglehub\datasets\sartajbhuvaji\brain-tumor-classification-mri\versions\3")
TRAIN_DIR = DATA_DIR / "Training"
TEST_DIR = DATA_DIR / "Testing"

# data split
VAL_SPLIT = 0.15  # 15% of training data for validation
RANDOM_STATE = 88  # for reproducible splits

# model classes
CLASSES = ['glioma', 'meningioma', 'pituitary', 'no_tumor']
NUM_CLASSES = len(CLASSES)

# training hyperparameters
BATCH_SIZE = 32  
NUM_WORKERS = 4 
PIN_MEMORY = True  
PERSISTENT_WORKERS = True 

# training phases
STAGE1_EPOCHS = 50  # initial training with frozen backbone
STAGE2_EPOCHS = 30  # fine-tuning with unfrozen backbone

# optimization
OPTIMIZER = 'adamw'  
LR_STAGE1 = 1e-3     # initial learning rate
LR_STAGE2 = 1e-4     # reduced learning rate for fine-tuning
WEIGHT_DECAY = 0.05  # weight decay for AdamW

# model architecture
MODEL_NAME = 'efficientnet_v2_s'  
PRETRAINED = True                # use pre-trained weights
FREEZE_BACKBONE = True           # freeze backbone during stage 1
DROPOUT_RATE = 0.3              # dropout in classifier head

# normalization
IMAGE_SIZE = 384  
MEAN = [0.485, 0.456, 0.406]    # imageNet mean
SD = [0.229, 0.224, 0.225]     # imageNet sd
AUTOAUGMENT = True             
MIXUP_ALPHA = 0.2               
CUTMIX_ALPHA = 1.0            

# mixed precision training
MIXED_PRECISION = True  
GRAD_SCALER = True      # use GradScaler for mixed precision

# lr Scheduling
LR_SCHEDULER = 'cosine' 
WARMUP_EPOCHS = 5        # warmup period
MIN_LR = 1e-6            # minimum learning rate

# stoppage
PATIENCE = 10            # patience for early stopping
MIN_DELTA = 0.001        # minimum change to qualify as improvement

# checkpointing
CHECKPOINT_DIR = BASE_DIR / 'checkpoints'
CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)

# Grad-CAM 
GRADCAM_LAYER = 'features.8'  
GRADCAM_DIR = BASE_DIR / 'reports' / 'gradcam'
GRADCAM_DIR.mkdir(exist_ok=True, parents=True)

# metrics
METRICS = ['accuracy', 'f1_score', 'precision', 'recall']
CONFUSION_MATRIX = True  
CLASSIFICATION_REPORT = True 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = BASE_DIR / "models" / "brain_tumor_model.pth"
GRAPHICS_DIR = BASE_DIR / "reports"

# ensure directories exist
MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
GRAPHICS_DIR.mkdir(parents=True, exist_ok=True)

# seed for reproducibility
torch.manual_seed(88)
np.random.seed(88)
random.seed(88)
if DEVICE.type == "cuda":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False