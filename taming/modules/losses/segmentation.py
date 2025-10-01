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

# class FocalBCELossWithQuant(nn.Module):
#     def __init__(self, codebook_weight=1., gamma=2.0, alpha=1.0):
#         super().__init__()
#         self.codebook_weight = codebook_weight
#         self.gamma = gamma
#         self.alpha = alpha

#     def forward(self, qloss, target, prediction, split):
#         bce_loss = F.binary_cross_entropy_with_logits(prediction, target, reduction='none')
#         prob = torch.sigmoid(prediction)
#         pt = prob * target + (1 - prob) * (1 - target)
#         focal_weight = self.alpha * (1 - pt) ** self.gamma
#         focal_bce = (focal_weight * bce_loss).mean()

#         loss = focal_bce + self.codebook_weight * qloss
#         return loss, {
#             "{}/total_loss".format(split): loss.clone().detach().mean(),
#             "{}/focal_bce_loss".format(split): focal_bce.detach().mean(),
#             "{}/quant_loss".format(split): qloss.detach().mean()
#         }