import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, target):
        smooth = 1e-5

        tp = torch.sum(inputs * target)
        fp = torch.sum(inputs) - tp
        fn = torch.sum(target) - tp

        dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)

        return 1 - dice
    
class TverskyLoss(nn.Module):
    def __init__(self, alpha = 0.3, beta = 0.7):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, inputs, target, reduction='mean'):
        smooth = 1e-5

        tp = torch.sum(inputs * target, dim=(0, 2, 3))
        fp = torch.sum(inputs) - tp
        fn = torch.sum(target) - tp

        tversky = (2 * tp + smooth) / ((2 * tp) + (self.alpha * fp) + (self.beta * fn) + smooth)
        loss = 1 - tversky
        if reduction == 'mean':
            return torch.mean(loss)
        elif reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss
    
class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha = 0.3, beta = 0.7, gamma = 3/4):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, inputs, target, reduction='mean'):
        smooth = 1e-5

        tp = torch.sum(inputs * target, dim=(0, 2, 3))
        fp = torch.sum(inputs) - tp
        fn = torch.sum(target) - tp

        tversky = (2 * tp + smooth) / ((2 * tp) + (self.alpha * fp) + (self.beta * fn) + smooth)
        loss = (1 - tversky).pow(self.gamma)
        if reduction == 'mean':
            return torch.mean(loss)
        elif reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss
        

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2): 
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets): 
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p = torch.sigmoid(inputs)
        pt = p * targets + (1 - p) * (1 - targets)
        F_loss = (1-pt)**self.gamma * BCE_loss
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            F_loss = alpha_t * F_loss
        return torch.mean(F_loss)
    
class HybridFocalLoss(nn.Module):
    def __init__(self, lmbda) -> None:
        super(HybridFocalLoss, self).__init__()
        self.lmbda = lmbda
        self.focal_loss = FocalLoss()
        self.focal_tversky_loss = FocalTverskyLoss()

    def forward(self, inputs, target, reduction='mean'):
        loss = self.lmbda * self.focal_loss(inputs, target) + (1 - self.lmbda) * self.focal_tversky_loss(inputs, target)
        if reduction == 'mean':
            return torch.mean(loss)
        elif reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss

class UnifiedFocalLoss(nn.Module):
    def __init__(self, delta, gamma, lmbda):
        super(UnifiedFocalLoss, self).__init__()
        self.delta = delta
        self.gamma = gamma
        self.lmbda = lmbda
        self.focal_loss = FocalLoss(alpha=delta, gamma=gamma)
        self.focal_tversky_loss = FocalTverskyLoss(alpha=delta, beta=1-delta, gamma=gamma)

    def forward(self, inputs, target, reduction='mean'):
        loss = self.lmbda * self.focal_loss(inputs, target) + (1 - self.lmbda) * self.focal_tversky_loss(inputs, target)
        if reduction == 'mean':
            return torch.mean(loss)
        elif reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss