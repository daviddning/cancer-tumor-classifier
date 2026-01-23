"""
attention U-Net for Brain Tumor Segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import config


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


class ConvBlock(nn.Module):
    """
    convolutional block: Conv -> BatchNorm -> Activation -> Conv -> BatchNorm -> Activation
    standard U-Net building block with configurable activation and dropout
    """
    def __init__(self, in_channels, out_channels, dropout_rate=None):
        super(ConvBlock, self).__init__()
        
        if dropout_rate is None:
            dropout_rate = config.DROPOUT_RATE
        
        layers = []
        
        # first conv block
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, 
                               bias=not config.USE_BATCH_NORM))
        if config.USE_BATCH_NORM:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(get_activation(config.ACTIVATION))
        
        if dropout_rate > 0:
            layers.append(nn.Dropout2d(dropout_rate))
        
        # second conv block
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, 
                               bias=not config.USE_BATCH_NORM))
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
    attention gate for skip connections
    helps model focus on relevant regions
    """
    def __init__(self, F_g, F_l, F_int):
        """
        Args:
            F_g: Number of feature channels from gating signal
            F_l: Number of feature channels from skip connection
            F_int: Number of intermediate channels
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
        Args:
            g: Gating signal from decoder (lower resolution)
            x: Skip connection from encoder (higher resolution)
        Returns:
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


class AttentionUNet(nn.Module):
    """
    U-Net with Attention Gates for Brain Tumor Segmentation
    
    Architecture:
    - Input: 4 channels (FLAIR, T1, T1ce, T2) at 240x240
    - Encoder: 4 downsampling blocks (64 -> 128 -> 256 -> 512)
    - Bottleneck: 1024 channels
    - Decoder: 4 upsampling blocks with attention gates
    - Output: 1 channel (binary tumor mask) at 240x240 with logits
    
    Key features:
    - Attention gates help focus on small tumors
    - Skip connections preserve spatial information
    - Outputs raw logits (no sigmoid in forward pass)
    """
    
    def __init__(self, in_channels=4, num_classes=1, base_filters=64):
        """
        Args:
            in_channels: Number of input channels (default: 4)
            num_classes: Number of output classes (default: 1)
            base_filters: Number of filters in first layer (default: 64)
        """
        super(AttentionUNet, self).__init__()
        
        # calculate filter sizes for each level
        filters = [base_filters * (2**i) for i in range(5)]  # [64, 128, 256, 512, 1024]
        
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
        self.up4 = nn.ConvTranspose2d(filters[4], filters[3], kernel_size=2, stride=2)
        self.att4 = AttentionGate(F_g=filters[3], F_l=filters[3], 
                                  F_int=filters[3]//config.ATTENTION_REDUCTION)
        self.dec4 = ConvBlock(filters[4], filters[3])  # 1024 -> 512
        
        self.up3 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2)
        self.att3 = AttentionGate(F_g=filters[2], F_l=filters[2], 
                                  F_int=filters[2]//config.ATTENTION_REDUCTION)
        self.dec3 = ConvBlock(filters[3], filters[2])  # 512 -> 256
        
        self.up2 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2)
        self.att2 = AttentionGate(F_g=filters[1], F_l=filters[1], 
                                  F_int=filters[1]//config.ATTENTION_REDUCTION)
        self.dec2 = ConvBlock(filters[2], filters[1])  # 256 -> 128
        
        self.up1 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2)
        self.att1 = AttentionGate(F_g=filters[0], F_l=filters[0], 
                                  F_int=filters[0]//config.ATTENTION_REDUCTION)
        self.dec1 = ConvBlock(filters[1], filters[0])  # 128 -> 64
        
        # output layer 
        self.out = nn.Conv2d(filters[0], num_classes, kernel_size=1)
    
    def forward(self, x, return_attention=False):
        """
        forward pass through the network
        
        Args:
            x: Input tensor of shape (batch, 4, 240, 240)
            return_attention: If True, return attention maps for visualization
        
        Returns:
            Output tensor of shape (batch, 1, 240, 240) with raw logits
            Optionally returns attention maps if return_attention=True
        """
        # encoder
        e1 = self.enc1(x)              # (B, 64, 240, 240)
        e2 = self.enc2(self.pool(e1))  # (B, 128, 120, 120)
        e3 = self.enc3(self.pool(e2))  # (B, 256, 60, 60)
        e4 = self.enc4(self.pool(e3))  # (B, 512, 30, 30)
        
        # bottleneck
        b = self.bottleneck(self.pool(e4))  # (B, 1024, 15, 15)
        
        # decoder
        d4 = self.up4(b)                    # (B, 512, 30, 30)
        e4_att, att_map4 = self.att4(d4, e4)
        d4 = torch.cat([d4, e4_att], dim=1) # (B, 1024, 30, 30)
        d4 = self.dec4(d4)                  # (B, 512, 30, 30)
        
        d3 = self.up3(d4)                   # (B, 256, 60, 60)
        e3_att, att_map3 = self.att3(d3, e3)
        d3 = torch.cat([d3, e3_att], dim=1) # (B, 512, 60, 60)
        d3 = self.dec3(d3)                  # (B, 256, 60, 60)
        
        d2 = self.up2(d3)                   # (B, 128, 120, 120)
        e2_att, att_map2 = self.att2(d2, e2)
        d2 = torch.cat([d2, e2_att], dim=1) # (B, 256, 120, 120)
        d2 = self.dec2(d2)                  # (B, 128, 120, 120)
        
        d1 = self.up1(d2)                   # (B, 64, 240, 240)
        e1_att, att_map1 = self.att1(d1, e1)
        d1 = torch.cat([d1, e1_att], dim=1) # (B, 128, 240, 240)
        d1 = self.dec1(d1)                  # (B, 64, 240, 240)
        
        out = self.out(d1)                  # (B, 1, 240, 240)
        
        if return_attention:
            attention_maps = {
                'att1': att_map1,
                'att2': att_map2,
                'att3': att_map3,
                'att4': att_map4
            }
            return out, attention_maps
        
        return out


def count_parameters(model):
    """count trainable parameters in model"""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total


def test_model(model, input_shape=(1, 4, 240, 240)):
    """
    test model with dummy input to verify architecture
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape (batch, channels, height, width)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # create dummy input
    x = torch.randn(input_shape).to(device)
    
    # forward pass
    print(f"\ntesting model with input shape: {input_shape}")
    
    with torch.no_grad():
        output = model(x, return_attention=False)
    
    print(f"total parameters: {count_parameters(model):,}")
    
    # calculate model size
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / (1024**2)
    print(f"model size: {size_mb:.2f} MB")
    
    # test attention maps
    print("\ntesting attention map output...")
    with torch.no_grad():
        output, attention_maps = model(x, return_attention=True)
    print(f"attention maps returned: {list(attention_maps.keys())}")
    for name, att_map in attention_maps.items():
        print(f"  {name}: {att_map.shape}")
    
    return output


if __name__ == "__main__":
    print("model verification")
    
    # Test model
    print("\ninitializing attention U-Net...")
    model = AttentionUNet(in_channels=4, num_classes=1, base_filters=config.BASE_FILTERS)
    
    print("\nrunning forward pass test...")
    output = test_model(model)
    
    print("ready for training!")
