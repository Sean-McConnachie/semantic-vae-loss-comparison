import torch
import torch.nn as nn
import torch.nn.functional as F


class BCELoss(nn.Module):
    def forward(self, prediction, target):
        loss = F.binary_cross_entropy_with_logits(prediction,target)
        return loss, {}


class BCELossWithQuant(nn.Module):
    def __init__(self, codebook_weight=1.0):
        super().__init__()
        self.codebook_weight = codebook_weight

    def forward(self, qloss, target, prediction, split):
        bce_loss = F.binary_cross_entropy_with_logits(prediction,target)
        loss = bce_loss + self.codebook_weight*qloss
        log_dict = {
            "{}/total_loss".format(split): loss.clone().detach().mean(),
            "{}/bce_loss".format(split): bce_loss.detach().mean(),
            "{}/quant_loss".format(split): qloss.detach().mean()
        }
        return loss, log_dict

class CELossWithQuant(nn.Module):
    def __init__(self, codebook_weight=1.0):
        super().__init__()
        self.codebook_weight = codebook_weight

    def forward(self, qloss, target, prediction, split):
        target_indices = torch.argmax(target, dim=1)
        ce_loss = F.cross_entropy(prediction, target_indices)
        loss = ce_loss + self.codebook_weight * qloss
        log_dict = {
            f"{split}/total_loss": loss.clone().detach().mean(),
            f"{split}/ce_loss": ce_loss.detach().mean(),
            f"{split}/quant_loss": qloss.detach().mean()
        }
        return loss, log_dict

class CEDiceLossWithQuant(nn.Module):
    def __init__(self, ce_weight=1.0, dice_weight=1.0, codebook_weight=1.0, num_classes=None, smooth=1e-6):
        super().__init__()
        if num_classes is None:
            raise ValueError("num_classes must be specified.")
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.codebook_weight = codebook_weight
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, qloss, prediction, target, split):
        if target.ndim == prediction.ndim and target.shape[1] == self.num_classes:
            target_indices = torch.argmax(target, dim=1)
        else:
            target_indices = target
        ce_loss = F.cross_entropy(prediction, target_indices)

        probs = F.softmax(prediction, dim=1)
        target_one_hot = F.one_hot(target_indices, num_classes=self.num_classes)
        target_one_hot = torch.moveaxis(target_one_hot, -1, 1).float()

        dice_score = 0.0
        for i in range(self.num_classes):
            p_class = probs[:, i]
            t_class = target_one_hot[:, i]

            intersection = (p_class * t_class).sum()
            union = p_class.sum() + t_class.sum()

            class_score = (2. * intersection + self.smooth) / (union + self.smooth)
            dice_score += class_score

        mean_dice_score = dice_score / self.num_classes
        dice_loss = 1 - mean_dice_score

        ce_dice_loss = (self.ce_weight * ce_loss) + (self.dice_weight * dice_loss)

        loss = ce_dice_loss + self.codebook_weight * qloss

        log_dict = {
            f"{split}/total_loss": loss.clone().detach().mean(),
            f"{split}/ce_loss": ce_loss.detach().mean(),
            f"{split}/dice_loss": dice_loss.detach().mean(),
            f"{split}/ce_dice_loss": ce_dice_loss.detach().mean(),
            f"{split}/quant_loss": qloss.detach().mean()
        }

        return loss, log_dict

class FocalLossWithQuant(nn.Module):
    def __init__(self, codebook_weight=1.0, gamma=2.0, alpha=None):
        super().__init__()
        self.codebook_weight = codebook_weight
        self.gamma = gamma

        if alpha is not None:
            if not isinstance(alpha, torch.Tensor):
                alpha = torch.tensor(alpha)
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None

    def forward(self, qloss, target, prediction, split):
        target_indices = torch.argmax(target, dim=1)
    
        ce_loss = F.cross_entropy(prediction, target_indices, weight=self.alpha, reduction='none')
    
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
    
        loss = focal_loss + self.codebook_weight * qloss
    
        log_dict = {
            f"{split}/total_loss": loss.clone().detach().mean(),
            f"{split}/focal_loss": focal_loss.detach().mean(),
            f"{split}/quant_loss": qloss.detach().mean()
        }
        return loss, log_dict
