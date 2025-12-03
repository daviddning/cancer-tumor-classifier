## grad-cam visualization utilities for training progress monitoring ##

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from typing import List, Tuple
from PIL import Image

from config import CLASSES, GRADCAM_CONFIG, DEVICE, MEAN, STD, GRADCAM_DIR


class GradCAM:
    def __init__(self, model, target_layer_name):
        
        ## initialize grad-cam for visualization ##

        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        
        # register hooks
        self._register_hooks()
    
    def _register_hooks(self):

        ## forward and backward hooks to capture activations and gradients ##
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # find and hook the target layer
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)
                break
    
    def generate_heatmap(self, input_tensor, target_class=None):
        
        ## generate grad-cam heatmap ##

        self.model.eval()
        
        with torch.enable_grad():
            # forward pass
            output = self.model(input_tensor)
            
            # determine target class
            if target_class is None:
                target_class = output.argmax(dim=1).item()
            
            # backward pass for target class
            self.model.zero_grad()
            target_score = output[0, target_class]
            target_score.backward()
        
        # compute grad-cam
        gradients = self.gradients[0] 
        activations = self.activations[0]  
        
        # weights by average gradient
        weights = gradients.mean(dim=(1, 2), keepdim=True) 
        heatmap = (weights * activations).sum(dim=0)  
        
        # ReLU
        heatmap = F.relu(heatmap)
        
        # normalize
        heatmap_min = heatmap.min()
        heatmap_max = heatmap.max()
        if heatmap_max > heatmap_min:
            heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
        
        return heatmap.cpu().numpy()


def denormalize_image(img_tensor):
    ## denormalize image for visualization ##
    img = img_tensor.clone().detach()
    for t, m, s in zip(img, MEAN, STD):
        t.mul_(s).add_(m)
    return torch.clamp(img, 0, 1)


def create_gradcam_visualization(
    model,
    dataloader,
    stage,
    epoch,
    num_samples=2,
    save_dir=None
):
    
    ## create and save grad-cam visualizations during training ##
     
    if save_dir is None:
        save_dir = GRADCAM_DIR
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # initialize grad-cam
    target_layer = GRADCAM_CONFIG['target_layers'][0]
    gradcam = GradCAM(model, target_layer)
    
    model.eval()
    
    # collect samples per class
    class_samples = {i: [] for i in range(len(CLASSES))}
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            for i, (img, label) in enumerate(zip(images, labels)):
                class_idx = label.item()
                if len(class_samples[class_idx]) < num_samples:
                    class_samples[class_idx].append(img.unsqueeze(0))
            
            if all(len(v) >= num_samples for v in class_samples.values()):
                break
    
    # create visualization
    fig, axes = plt.subplots(len(CLASSES), num_samples * 2, figsize=(15, 10))
    fig.suptitle(f'{stage.upper()} - epoch {epoch+1}: grad-cam heatmaps', fontsize=14, fontweight='bold')
    
    for class_idx, samples in class_samples.items():
        for sample_idx, img_tensor in enumerate(samples[:num_samples]):
            img_tensor = img_tensor.to(DEVICE)
            
            # generate heatmap
            heatmap = gradcam.generate_heatmap(img_tensor, target_class=class_idx)
            
            # denormalize image
            img_display = denormalize_image(img_tensor[0]).permute(1, 2, 0).cpu().numpy()
            
            # original image
            ax = axes[class_idx, sample_idx * 2]
            ax.imshow(img_display)
            ax.set_title(f'{CLASSES[class_idx]}')
            ax.axis('off')
            
            # heatmap overlay
            ax = axes[class_idx, sample_idx * 2 + 1]
            im = ax.imshow(img_display)
            heatmap_colored = cm.jet(heatmap)
            ax.imshow(heatmap_colored, alpha=GRADCAM_CONFIG['alpha'], cmap='jet')
            ax.set_title(f'{CLASSES[class_idx]} (GRAD-CAM)')
            ax.axis('off')
    
    plt.tight_layout()
    
    # save
    save_path = save_dir / f'{stage}_epoch_{epoch+1}_gradcam.png'
    plt.savefig(save_path, dpi=GRADCAM_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    
    print(f"grad-cam visualization saved to {save_path}")
