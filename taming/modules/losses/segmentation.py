import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse as ifilterfalse


from main import get_obj_from_str
from taming.modules.losses.vqperceptual import vanilla_d_loss, hinge_d_loss, adopt_weight
from taming.modules.discriminator.model import NLayerDiscriminator, weights_init

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
