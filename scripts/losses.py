"""
loss Functions for segmentation model
dice Loss, BCE Loss, and focal Loss 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import config

# dice loss
class DiceLoss(nn.Module):
    """
    dice Loss for segmentation tasks
    measures overlap between prediction and ground truth
    """
    
    def __init__(self, smooth=1.0):
        """
        args:
            smooth: smoothing factor to avoid division by zero
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        args:
            pred: predicted mask (B, 1, H, W) with values in [0, 1]
            target: ground truth mask (B, 1, H, W) with values in {0, 1}
        returns:
            dice loss value
        """
        # flatten predictions and targets
        pred = pred.view(-1)
        target = target.view(-1)
        
        # calculate intersection and union
        intersection = (pred * target).sum()
        dice_score = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        # feturn loss 
        return 1 - dice_score



# focal loss
class FocalLoss(nn.Module):
    """
    focal loss for handling hard examples
    focuses training on difficult-to-classify pixels
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        args:
            alpha: weighting factor for positive class
            gamma: focusing parameter
            reduction: 'mean' or 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred, target):
        """
        args:
            pred: predicted mask (B, 1, H, W) with values in [0, 1]
            target: ground truth mask (B, 1, H, W) with values in {0, 1}
        returns:
            focal loss value
        """
        # flatten
        pred = pred.view(-1)
        target = target.view(-1)
        
        # calculate BCE
        bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        
        # calculate focal weight
        pt = torch.exp(-bce_loss)  
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        
        # apply focal weight
        focal_loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss



# binary cross entropy loss
class BCELoss(nn.Module):
    """
    bce loss
    standard loss for binary segmentation
    """
    
    def __init__(self, pos_weight=None):
        """
        args:
            pos_weight: weight for positive class
        """
        super(BCELoss, self).__init__()
        self.pos_weight = pos_weight
    
    def forward(self, pred, target):
        if self.pos_weight is not None:
            return F.binary_cross_entropy(pred, target, 
                                        weight=torch.tensor([self.pos_weight]).to(pred.device),
                                        reduction='mean')
        else:
            return F.binary_cross_entropy(pred, target, reduction='mean')

# COMBINED LOSS
class CombinedLoss(nn.Module):
    """
    combined Loss: weighted sum of Dice, BCE, and Focal losses
    """
    
    def __init__(
        self, 
        dice_weight=config.DICE_WEIGHT,
        bce_weight=config.BCE_WEIGHT,
        focal_weight=config.FOCAL_WEIGHT,
        pos_weight=config.POS_WEIGHT
    ):
        """
        args:
            dice_weight: weight for dice loss (default: 0.5)
            bce_weight: weight for bce loss (default: 0.3)
            focal_weight: weight for focal loss (default: 0.2)
            pos_weight: positive class weight for bce (default: 1.3)
        """
        super(CombinedLoss, self).__init__()
        
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        
        # Initialize loss functions
        self.dice_loss = DiceLoss(smooth=1.0)
        self.bce_loss = BCELoss(pos_weight=pos_weight)
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        
        print(f"\ncombined Loss initialized:")
        print(f"  dice weight: {dice_weight}")
        print(f"  bce weight: {bce_weight}")
        print(f"  focal weight: {focal_weight}")
        print(f"  positive class weight: {pos_weight}")
    
    def forward(self, pred, target):
        """
        args:
            pred: predicted mask (B, 1, H, W) with values in [0, 1]
            target: ground truth mask (B, 1, H, W) with values in {0, 1}
        returns:
            combined loss value and individual loss components
        """
        # calculate individual losses
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target)
        focal = self.focal_loss(pred, target)
        
        # weighted combination
        total_loss = (
            self.dice_weight * dice + 
            self.bce_weight * bce + 
            self.focal_weight * focal
        )
        
        # return total loss and components 
        return total_loss, {
            'dice_loss': dice.item(),
            'bce_loss': bce.item(),
            'focal_loss': focal.item(),
            'total_loss': total_loss.item()
        }


# loss utilities 
def get_loss_function(loss_type='combined'):
    """
    factory function to get loss function by name
    
    args:
        loss_type: 'dice', 'bce', 'focal', or 'combined'
    returns:
        loss function instance
    """
    if loss_type == 'dice':
        return DiceLoss()
    elif loss_type == 'bce':
        return BCELoss(pos_weight=config.POS_WEIGHT)
    elif loss_type == 'focal':
        return FocalLoss()
    elif loss_type == 'combined':
        return CombinedLoss()
    else:
        raise ValueError(f"unknown loss type: {loss_type}")


def test_loss_functions():
    """test all loss functions with dummy data"""
    print("testing loss functions")

    # create dummy predictions and targets
    batch_size = 4
    pred = torch.rand(batch_size, 1, 240, 240)  
    target = torch.randint(0, 2, (batch_size, 1, 240, 240)).float()  
    
    print(f"\ninput shapes:")
    print(f"  predictions: {pred.shape}")
    print(f"  targets: {target.shape}")
    print(f"  prediction range: [{pred.min():.3f}, {pred.max():.3f}]")
    print(f"  target values: {target.unique().tolist()}")
    
    # test dice loss
    print("\n1. dice loss")
    dice_loss = DiceLoss()
    dice_value = dice_loss(pred, target)
    print(f"dice loss: {dice_value.item():.4f}")
    
    # test bce loss
    print("\n2. bce loss")
    bce_loss = BCELoss(pos_weight=config.POS_WEIGHT)
    bce_value = bce_loss(pred, target)
    print(f"bce loss: {bce_value.item():.4f}")
    
    # test focal loss
    print("\n3. focal loss")
    focal_loss = FocalLoss()
    focal_value = focal_loss(pred, target)
    print(f"focal loss: {focal_value.item():.4f}")
    
    # test Combined Loss
    print("\n4. combined loss")
    combined_loss = CombinedLoss()
    total_loss, components = combined_loss(pred, target)
    print(f"\nloss components:")
    print(f"  dice Loss:  {components['dice_loss']:.4f}")
    print(f"  bce Loss:   {components['bce_loss']:.4f}")
    print(f"  focal Loss: {components['focal_loss']:.4f}")
    print(f"  total Loss: {components['total_loss']:.4f}")
    print("\ncombined loss ready for training")
    
    # test gradient flow
    print("\n5. gradient flow test")
    pred.requires_grad = True
    loss, _ = combined_loss(pred, target)
    loss.backward()
    print(f"gradient computed: {pred.grad is not None}")
    print(f"gradient shape: {pred.grad.shape}")
    print(f"gradient range: [{pred.grad.min():.6f}, {pred.grad.max():.6f}]")
    print("gradients flowing correctly")
    
    print("\nall loss functions working correctly")
    print("ready for training!")
    print("combined loss will be used by default")


# main
if __name__ == "__main__":
    
    # test all loss functions
    test_loss_functions()