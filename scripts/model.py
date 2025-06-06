import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from config import NUM_CLASSES

def get_efficientnet_v2_small(pretrained=True):
    weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
    model = efficientnet_v2_s(weights=weights)
    # replace final classifier
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
    return model

def unfreeze_last_n_blocks(model, n=3):
    children = list(model.features.children())
    for i, layer in enumerate(children):
        if i >= len(children) - n:
            for param in layer.parameters():
                param.requires_grad = True
        else:
            for param in layer.parameters():
                param.requires_grad = False