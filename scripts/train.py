"""
Attention U-Net with Grad-CAM for Brain Tumor Segmentation.
Includes Grad-CAM visualization to monitor what the model is learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import config

# activation helper
def get_activation(activation_type):
    """returns activation function based on string"""
    if activation_type == 'relu':
        return nn.ReLU(inplace=True)
    elif activation_type == 'leaky_relu':
        return nn.LeakyReLU(0.01, inplace=True)
    elif activation_type == 'elu':
        return nn.ELU(inplace=True)
    else:
        return nn.ReLU(inplace=True)

# blocks
class ConvBlock(nn.Module):
    """
    convolutional block: Conv -> BatchNorm -> Activation -> Conv -> BatchNorm -> Activation
    standard U-Net building block with configurable activation and dropout
    """
    def __init__(self, in_channels, out_channels, dropout_rate=config.DROPOUT_RATE):
        super(ConvBlock, self).__init__()
        
        layers = []
        
        # first conv block
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not config.USE_BATCH_NORM))
        if config.USE_BATCH_NORM:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(get_activation(config.ACTIVATION))
        
        if dropout_rate > 0:
            layers.append(nn.Dropout2d(dropout_rate))
        
        # second conv block
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=not config.USE_BATCH_NORM))
        if config.USE_BATCH_NORM:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(get_activation(config.ACTIVATION))
        
        if dropout_rate > 0:
            layers.append(nn.Dropout2d(dropout_rate))
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.conv(x)


class AttentionGate(nn.Module):
    """
    attention Gate for skip connections
    helps model focus on relevant regions 
    """
    def __init__(self, F_g, F_l, F_int):
        """
        args:
            F_g: number of feature channels from gating signal
            F_l: number of feature channels from skip connection
            F_int: number of intermediate channels
        """
        super(AttentionGate, self).__init__()
        
        # transform gating signal 
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int) if config.USE_BATCH_NORM else nn.Identity()
        )
        
        # transform skip connection
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int) if config.USE_BATCH_NORM else nn.Identity()
        )
        
        # generate attention coefficients
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1) if config.USE_BATCH_NORM else nn.Identity(),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        """
        args:
            g: Gating signal from decoder (lower resolution)
            x: Skip connection from encoder (higher resolution)
        returns:
            Attention-weighted skip connection and attention map
        """
        # apply transformations
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # combine and generate attention weights
        psi = self.relu(g1 + x1)
        attention_map = self.psi(psi)
        
        # apply attention weights to skip connection
        return x * attention_map, attention_map


# u-net model
class AttentionUNet(nn.Module):
    """
    U-Net with Attention Gates and Grad-CAM support
    
    architecture:
    - input: 4 channels (FLAIR, T1, T1ce, T2) at 240x240
    - encoder: 4 downsampling blocks (64 -> 128 -> 256 -> 512)
    - bottleneck: 1024 channels
    - decoder: 4 upsampling blocks with attention gates
    - output: 1 channel (binary tumor mask) at 240x240
    
    key features:
    - attention gates help focus on small tumors
    - skip connections preserve spatial information
    - grad-CAM support for visualization
    - configurable hyperparameters
    """
    
    def __init__(self, in_channels=4, num_classes=1, base_filters=config.BASE_FILTERS):
        """
        args:
            in_channels: number of input channels (4)
            num_classes: number of output classes (1)
            base_filters: number of filters in first layer (scales by 2 each level)
        """
        super(AttentionUNet, self).__init__()
        
        # calculate filter sizes for each level
        filters = [base_filters * (2**i) for i in range(5)]  # [64, 128, 256, 512, 1024]
        
        # store features for Grad-CAM
        self.gradients = None
        self.activations = None
        self.attention_maps = {}
        
        # encoder
        self.enc1 = ConvBlock(in_channels, filters[0])      # 4 -> 64
        self.enc2 = ConvBlock(filters[0], filters[1])       # 64 -> 128
        self.enc3 = ConvBlock(filters[1], filters[2])       # 128 -> 256
        self.enc4 = ConvBlock(filters[2], filters[3])       # 256 -> 512
        
        # max pooling for downsampling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # bottleneck
        self.bottleneck = ConvBlock(filters[3], filters[4])  # 512 -> 1024
        
        # decoder
        
        # level 4
        self.up4 = nn.ConvTranspose2d(filters[4], filters[3], kernel_size=2, stride=2)
        self.att4 = AttentionGate(F_g=filters[3], F_l=filters[3], F_int=filters[3]//config.ATTENTION_REDUCTION)
        self.dec4 = ConvBlock(filters[4], filters[3])  # 1024 -> 512
        
        # level 3
        self.up3 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2)
        self.att3 = AttentionGate(F_g=filters[2], F_l=filters[2], F_int=filters[2]//config.ATTENTION_REDUCTION)
        self.dec3 = ConvBlock(filters[3], filters[2])  # 512 -> 256
        
        # level 2
        self.up2 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2)
        self.att2 = AttentionGate(F_g=filters[1], F_l=filters[1], F_int=filters[1]//config.ATTENTION_REDUCTION)
        self.dec2 = ConvBlock(filters[2], filters[1])  # 256 -> 128
        
        # level 1
        self.up1 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2)
        self.att1 = AttentionGate(F_g=filters[0], F_l=filters[0], F_int=filters[0]//config.ATTENTION_REDUCTION)
        self.dec1 = ConvBlock(filters[1], filters[0])  # 128 -> 64
        
        # output layer
        self.out = nn.Conv2d(filters[0], num_classes, kernel_size=1)
    
    def forward(self, x, return_attention=False):
        """
        forward pass through the network
        
        args:
            x: input tensor of shape (batch, 4, 240, 240)
            return_attention: if True, return attention maps for visualization
        returns:
            output tensor of shape (batch, 1, 240, 240) with values in [0, 1]
        """
        # encoder
        e1 = self.enc1(x)              # (B, 64, 240, 240)
        e2 = self.enc2(self.pool(e1))  # (B, 128, 120, 120)
        e3 = self.enc3(self.pool(e2))  # (B, 256, 60, 60)
        e4 = self.enc4(self.pool(e3))  # (B, 512, 30, 30)
        
        # bottleneck
        b = self.bottleneck(self.pool(e4))  # (B, 1024, 15, 15)
        
        # register hook for Grad-CAM
        if config.GRADCAM_LAYER == 'bottleneck' and b.requires_grad:
            b.register_hook(self.save_gradient)
            self.activations = b
        
        # decoder
        
        # level 4
        d4 = self.up4(b)                    # (B, 512, 30, 30)
        e4_att, att_map4 = self.att4(d4, e4)  
        d4 = torch.cat([d4, e4_att], dim=1) # (B, 1024, 30, 30)
        d4 = self.dec4(d4)                  # (B, 512, 30, 30)
        
        if return_attention:
            self.attention_maps['att4'] = att_map4
        
        if config.GRADCAM_LAYER == 'dec4' and d4.requires_grad:
            d4.register_hook(self.save_gradient)
            self.activations = d4
        
        # level 3
        d3 = self.up3(d4)                   # (B, 256, 60, 60)
        e3_att, att_map3 = self.att3(d3, e3)  
        d3 = torch.cat([d3, e3_att], dim=1) # (B, 512, 60, 60)
        d3 = self.dec3(d3)                  # (B, 256, 60, 60)
        
        if return_attention:
            self.attention_maps['att3'] = att_map3
        
        if config.GRADCAM_LAYER == 'dec3' and d3.requires_grad:
            d3.register_hook(self.save_gradient)
            self.activations = d3
        
        # level 2
        d2 = self.up2(d3)                   # (B, 128, 120, 120)
        e2_att, att_map2 = self.att2(d2, e2)  
        d2 = torch.cat([d2, e2_att], dim=1) # (B, 256, 120, 120)
        d2 = self.dec2(d2)                  # (B, 128, 120, 120)
        
        if return_attention:
            self.attention_maps['att2'] = att_map2
        
        if config.GRADCAM_LAYER == 'dec2' and d2.requires_grad:
            d2.register_hook(self.save_gradient)
            self.activations = d2
        
        # level 1
        d1 = self.up1(d2)                   # (B, 64, 240, 240)
        e1_att, att_map1 = self.att1(d1, e1)  
        d1 = torch.cat([d1, e1_att], dim=1) # (B, 128, 240, 240)
        d1 = self.dec1(d1)                  # (B, 64, 240, 240)
        
        if return_attention:
            self.attention_maps['att1'] = att_map1
        
        if config.GRADCAM_LAYER == 'dec1' and d1.requires_grad:
            d1.register_hook(self.save_gradient)
            self.activations = d1
        
        # output 
        out = self.out(d1)                  # (B, 1, 240, 240)
        
        if config.OUTPUT_ACTIVATION == 'sigmoid':
            out = torch.sigmoid(out)
        elif config.OUTPUT_ACTIVATION == 'softmax':
            out = torch.softmax(out, dim=1)
        
        if return_attention:
            return out, self.attention_maps
        
        return out
    
    def save_gradient(self, grad):
        """hook to save gradients for Grad-CAM"""
        self.gradients = grad


# grad-cam
class GradCAM:
    """
    grad-CAM 
    visualizes which regions the model focuses on for predictions
    """
    
    def __init__(self, model):
        self.model = model
        self.model.eval()
    
    def generate_cam(self, input_image, target_layer='bottleneck'):
        """
        generate Grad-CAM heatmap
        
        args:
            input_image: input tensor (1, 4, 240, 240)
            target_layer: layer to visualize
        returns:
            cam: grad-CAM heatmap (240, 240)
        """
        # set target layer
        global GRADCAM_LAYER
        GRADCAM_LAYER = target_layer
        
        # forward pass
        self.model.train()  # gradient computation
        output = self.model(input_image)
        
        # backward pass
        self.model.zero_grad()
        output.sum().backward()
        
        # get gradients and activations
        gradients = self.model.gradients
        activations = self.model.activations
        
        if gradients is None or activations is None:
            print(f"no gradients/activations captured for layer {target_layer}")
            return np.zeros((240, 240))
        
        # calculate weights 
        weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
        
        # weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        # apply ReLU 
        cam = F.relu(cam)
        
        # normalize to [0, 1]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        # resize to input size
        cam = F.interpolate(cam, size=(240, 240), mode='bilinear', align_corners=False)
        
        # convert to numpy
        cam = cam.squeeze().cpu().detach().numpy()
        
        self.model.eval()
        return cam
    
    def visualize_cam(self, input_image, cam, ground_truth=None, prediction=None, save_path=None):
        """
        visualize Grad-CAM overlay on input image
        
        args:
            input_image: input tensor (1, 4, 240, 240)
            cam: grad-CAM heatmap (240, 240)
            ground_truth: ground truth mask (optional)
            prediction: model prediction (optional)
            save_path: path to save figure
        """
        # use FLAIR channel for visualization
        flair = input_image[0, 0].cpu().numpy()
        
        # normalize FLAIR to [0, 1] for display
        flair = (flair - flair.min()) / (flair.max() - flair.min() + 1e-8)
        
        # create figure
        n_cols = 2 + (ground_truth is not None) + (prediction is not None)
        fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 5))
        
        col = 0
        
        # original image
        axes[col].imshow(flair, cmap='gray')
        axes[col].set_title('FLAIR Image')
        axes[col].axis('off')
        col += 1
        
        # grad-CAM overlay
        axes[col].imshow(flair, cmap='gray')
        axes[col].imshow(cam, cmap='jet', alpha=0.5)
        axes[col].set_title(f'Grad-CAM ({config.GRADCAM_LAYER})')
        axes[col].axis('off')
        col += 1
        
        # ground truth
        if ground_truth is not None:
            gt = ground_truth[0, 0].cpu().numpy()
            axes[col].imshow(flair, cmap='gray')
            axes[col].imshow(gt, cmap='jet', alpha=0.5)
            axes[col].set_title('Ground Truth')
            axes[col].axis('off')
            col += 1
        
        # prediction
        if prediction is not None:
            pred = prediction[0, 0].cpu().detach().numpy()
            axes[col].imshow(flair, cmap='gray')
            axes[col].imshow(pred, cmap='jet', alpha=0.5)
            axes[col].set_title('Prediction')
            axes[col].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        plt.close()


# utilities 
def count_parameters(model):
    """count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model(model, input_shape=(1, 4, 240, 240)):
    """
    test model with dummy input to verify architecture
    
    args:
        model: pyTorch model
        input_shape: input tensor shape (batch, channels, height, width)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # create dummy input
    x = torch.randn(input_shape).to(device)
    
    # forward pass
    print(f"\ntesting model with input shape: {input_shape}")
    print(f"device: {device}")
    
    with torch.no_grad():
        output = model(x, return_attention=False)
    
    print(f"output shape: {output.shape}")
    print(f"output range: [{output.min():.3f}, {output.max():.3f}]")
    print(f"total parameters: {count_parameters(model):,}")
    
    # calculate model size
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / (1024**2)
    print(f"model size: {size_mb:.2f} MB")
    
    return output


# main
if __name__ == "__main__":
    print("model verification")
        
    # test model
    print("\nmodel architecture test")
    model = AttentionUNet(in_channels=4, num_classes=1)
    test_model(model)
    
    # test Grad-CAM
    print("grad-cam test")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # create dummy input
    dummy_input = torch.randn(1, 4, 240, 240).to(device)
    dummy_gt = torch.randint(0, 2, (1, 1, 240, 240)).float().to(device)
    
    # generate Grad-CAM
    gradcam = GradCAM(model)
    cam = gradcam.generate_cam(dummy_input, target_layer='bottleneck')
    
    print(f"\ngrad-CAM generated successfully")
    print(f"  CAM shape: {cam.shape}")
    print(f"  CAM range: [{cam.min():.3f}, {cam.max():.3f}]")
    
    # visualize
    with torch.no_grad():
        pred = model(dummy_input)
    
    save_path = config.GRADCAM_SAVE_PATH / "test_gradcam.png"
    gradcam.visualize_cam(dummy_input, cam, ground_truth=dummy_gt, prediction=pred, save_path=save_path)
    print(f"\ngrad-CAM visualization saved to: {save_path}")
    
    print("model verification complete")
    print("\nmodel ready for training!")
    print("grad-CAM visualization working!")
