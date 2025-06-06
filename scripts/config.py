import os
from pathlib import Path
import torch
import numpy as np
import random

# configuration
BASE_DIR = Path(__file__).parent.parent.resolve()
TRAIN_DIR = BASE_DIR / "data" / "processed" / "training"
VAL_DIR = BASE_DIR / "data" / "processed" / "validation"
TEST_DIR = BASE_DIR / "data" / "processed" / "testing-1"
CLASSES = ['glioma', 'meningioma', 'pituitary']
NUM_CLASSES = len(CLASSES)

# hyperparameters
BATCH_SIZE = 24
STAGE1_EPOCHS = 15
STAGE2_EPOCHS = 20
LR_STAGE1 = 5e-4
LR_STAGE2 = 1.5e-4
WEIGHT_DECAY = 1e-5
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