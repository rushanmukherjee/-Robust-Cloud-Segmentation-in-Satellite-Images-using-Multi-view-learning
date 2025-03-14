import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedFocalDiceLoss(nn.Module):
    def __init__(self, lamda=0.25, alpha=1.0, gamma=2.0, beta=1.0, smooth=1e-6):
        """
        Initialize the combined Focal + Dice loss function.
        
        Parameters:
        - lamda: Weighting factor for class balancing in focal loss
        - alpha: Weighting factor for Focal Loss.
        - gamma: Focusing parameter for Focal Loss.
        - beta: Weighting factor for Dice Loss.
        - smooth: Smoothing factor to avoid division by zero in Dice Loss.
        """
        super(CombinedFocalDiceLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.smooth = smooth
        self.lamda = lamda

    def forward(self, logits, targets):
        """
        Compute the combined loss.

        Parameters:
        - logits: Predicted logits from the model (shape: [batch_size, num_classes, ...]).
        - targets: Ground truth labels (shape: [batch_size, num_classes, ...]).

        Returns:
        - Combined loss (scalar).
        """
        # Apply softmax to logits for multi-class segmentation
        epsilon = 1e-6
        probs = torch.softmax(logits, dim=1)
        probs = torch.clamp(probs, epsilon, 1.0 - epsilon)
        print(f"Probs shape: {probs.shape}")
        
        # One-hot encode targets if needed
        if len(targets.shape) == len(logits.shape) - 1:
            targets_one_hot = F.one_hot(targets, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
        else:
            targets_one_hot = targets.float()

        print(f"Targets shape: {targets_one_hot.shape}")
        
        # Compute Focal Loss
        cross_entropy = -targets_one_hot * torch.log(probs)
        focal_loss = self.lamda * torch.pow(1.0 - probs, self.gamma) * cross_entropy
        focal_loss = torch.mean(torch.sum(focal_loss, dim=(1, 2, 3)))
        print(f"Focal Loss: {focal_loss}")

        # focal_loss = -self.alpha * ((1 - probs) ** self.gamma) * targets_one_hot * torch.log(probs + self.smooth)
        # print(f"Focal Loss: {focal_loss}")
        # focal_loss = focal_loss.sum(dim=(1, 2, 3)).mean()
        # print(f"Focal Loss after mean: {focal_loss}")

        # Compute Dice Loss
        intersection = torch.sum(probs * targets_one_hot, dim=(1, 2, 3))
        union = torch.sum(targets_one_hot, dim=(1, 2, 3)) + torch.sum(probs, dim=(1, 2, 3))
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - torch.mean(dice_score)
        print(f"Dice Loss: {dice_loss}")
        
        # dice_loss = 1 - (2. * intersection + self.smooth) / (probs.sum(dim=(1, 2, 3)) + targets_one_hot.sum(dim=(1, 2, 3)) + self.smooth)
        # print(f"Dice Loss before mean: {dice_loss}")
        # dice_loss = dice_loss.mean()
        # print(f"Dice Loss after mean: {dice_loss}")

        # Combine losses
        combined_loss = self.alpha * focal_loss + self.beta * dice_loss
        return combined_loss

