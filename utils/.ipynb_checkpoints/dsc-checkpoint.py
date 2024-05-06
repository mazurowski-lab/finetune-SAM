import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):

        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)

def DICESEN_loss(input, target):
    smooth = 0.00000001
    y_true_f = input.view(-1)
    y_pred_f = target.view(-1)
    intersection = torch.sum(torch.mul(y_true_f,y_pred_f))
    dice= (2. * intersection ) / (torch.mul(y_true_f,y_true_f).sum() + torch.mul(y_pred_f,y_pred_f).sum() + smooth)
    sen = (1. * intersection ) / (torch.mul(y_true_f,y_true_f).sum() + smooth)
    return 2-dice-sen   

class DiceSensitivityLoss(nn.Module):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        super(DiceSensitivityLoss, self).__init__()
    
    def forward(self, inputs, targets, smooth = 1.):

        if self.n_classes == 1:
            inputs = torch.sigmoid(inputs)
        else:
            inputs = F.softmax(inputs, dim=1)

        y_true_f = inputs.view(-1)
        y_pred_f = targets.view(-1)

        intersection = (y_true_f * y_pred_f).sum()

        dice= (2. * intersection + smooth) / (y_pred_f.sum() + y_true_f.sum() + smooth)

        sen = (1. * intersection ) / (torch.mul(y_true_f,y_true_f).sum() + smooth)

        return 2 - dice-sen


def dice_coeff_multi_class(pred, target, n_classes):
    """Calculate the mean Dice Coefficient for multi-class data."""
    dice_scores = []
    for cls in range(n_classes):  # Iterate over each class
        pred_cls = (pred == cls).long()  # Binary mask for current predicted class
        target_cls = (target == cls).long()  # Binary mask for current actual class

        smooth = 1.0
        intersection = (pred_cls & target_cls).float().sum((1, 2))  # Sum over height and width dimensions
        union = pred_cls.float().sum((1, 2)) + target_cls.float().sum((1, 2))
        
        dice = (2. * intersection + smooth) / (union + smooth)
        dice_scores.append(dice)
    
    return torch.stack(dice_scores).mean() 