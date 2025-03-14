import torch.nn as nn
import torch
import torch.nn.functional as F

## Define Dice Loss Function
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

        self.smooth=1e-6

    def forward(self, logits, targets, smooth=1):    
        
        # Apply softmax to logits for multi-class segmentation
        epsilon = 1e-6
        probs = torch.softmax(logits, dim=1)
        probs = torch.clamp(probs, epsilon, 1.0 - epsilon)
        #print(f"Probs shape: {probs.shape}")

        # One-hot encode targets if needed
        if len(targets.shape) == len(logits.shape) - 1:
            targets_one_hot = F.one_hot(targets, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
        else:
            targets_one_hot = targets.float()

        #print(f"Targets shape: {targets_one_hot.shape}")

        intersection = torch.sum(probs * targets_one_hot, dim=(1, 2, 3))
        union = torch.sum(targets_one_hot, dim=(1, 2, 3)) + torch.sum(probs, dim=(1, 2, 3))
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - torch.mean(dice_score)
        #print(f"Dice Loss: {dice_loss}")

        return dice_loss
