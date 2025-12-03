## grad-cam visualization utilities ##

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
    ## grad-cam for model attention visualization ##
    
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        self._register_hooks()
    
    def _register_hooks(self):
        ## register forward and backward hooks ##
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                module.register_forward_hook(forward_hook)
                module.register_full_backward_hook(backward_hook)
                break
    
    def generate_heatmap(self, input_tensor, target_class=None):
        ## generate grad-cam heatmap ##
        
        self.model.eval()
        
        with torch.enable_grad():
            output = self.model(input_tensor)
            
            if target_class is None:
                target_class = output.argmax(dim=1).item()
            
            self.model.zero_grad()
            target_score = output[0, target_class]
            target_score.backward()
        
        ## compute weighted activation map ##
        gradients = self.gradients[0] 
        activations = self.activations[0]  
        
        weights = gradients.mean(dim=(1, 2), keepdim=True) 
        heatmap = (weights * activations).sum(dim=0)  
        
        heatmap = F.relu(heatmap)
        
        ## normalize to [0, 1] ##
        heatmap_min = heatmap.min()
        heatmap_max = heatmap.max()
        if heatmap_max > heatmap_min:
            heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
        
        return heatmap.cpu().numpy()


def denormalize_image(img_tensor):
    ## denormalize image tensor for visualization ##
    
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
    ## create multi-layer grad-cam visualizations ##
    
    if save_dir is None:
        save_dir = GRADCAM_DIR
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    
    ## collect samples per class ##
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
    
    ## create visualization for each target layer ##
    target_layers = GRADCAM_CONFIG['target_layers']
    
    for layer_name in target_layers:
        gradcam = GradCAM(model, layer_name)
        
        fig, axes = plt.subplots(
            len(CLASSES), 
            num_samples * 2, 
            figsize=GRADCAM_CONFIG['figure_size']
        )
        fig.suptitle(
            f'{stage.upper()} epoch {epoch+1}: grad-cam ({layer_name})', 
            fontsize=14, 
            fontweight='bold'
        )
        
        for class_idx, samples in class_samples.items():
            for sample_idx, img_tensor in enumerate(samples[:num_samples]):
                img_tensor = img_tensor.to(DEVICE)
                
                ## generate heatmap ##
                heatmap = gradcam.generate_heatmap(img_tensor, target_class=class_idx)
                
                ## denormalize ##
                img_display = denormalize_image(img_tensor[0]).permute(1, 2, 0).cpu().numpy()
                
                ## original image ##
                ax = axes[class_idx, sample_idx * 2]
                ax.imshow(img_display)
                ax.set_title(f'{CLASSES[class_idx]}')
                ax.axis('off')
                
                ## heatmap overlay ##
                ax = axes[class_idx, sample_idx * 2 + 1]
                ax.imshow(img_display)
                heatmap_colored = cm.jet(heatmap)
                ax.imshow(heatmap_colored, alpha=GRADCAM_CONFIG['alpha'], cmap='jet')
                ax.set_title(f'{CLASSES[class_idx]} (grad-cam)')
                ax.axis('off')
        
        plt.tight_layout()
        
        ## save ##
        layer_safe_name = layer_name.replace('.', '_')
        save_path = save_dir / f'{stage}_epoch_{epoch+1}_{layer_safe_name}_gradcam.png'
        plt.savefig(save_path, dpi=GRADCAM_CONFIG['dpi'], bbox_inches='tight')
        plt.close()
        
        print(f"  grad-cam saved: {save_path.name}")