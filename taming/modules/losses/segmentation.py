import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def forward(self, qloss, target, prediction, split):
        target_indices = torch.argmax(target, dim=1)
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
        print(f"FocalLossWithQuant running with gamma={self.gamma} and alpha={self.alpha}")

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


class BaseLossWithDiscriminator(nn.Module):
    def __init__(
        self,
        disc_start,
        disc_num_layers=3,
        disc_in_channels=183,
        disc_factor=1.0,
        disc_weight=1.0,
        base_loss_target="taming.modules.losses.segmentation.CELossWithQuant",
        base_loss_params={},
        base_loss_weight=1.0,
        use_actnorm=False,
        disc_conditional=False,
        disc_ndf=64,
        disc_loss="hinge"
    ):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.base_loss = get_obj_from_str(base_loss_target)(**base_loss_params)
        self.base_loss_weight = base_loss_weight

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm,
                                                 ndf=disc_ndf
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(
        self,
        codebook_loss,
        inputs,
        reconstructions,
        optimizer_idx,
        global_step,
        last_layer=None,
        cond=None,
        split="train"
    ):
        if self.base_loss_weight > 0:
            # Assumes base_loss(qloss, target, prediction, split)
            base_loss, base_loss_logs = self.base_loss(codebook_loss, inputs.contiguous(), reconstructions.contiguous(), split)
            base_loss = self.base_loss_weight * base_loss
        else:
            base_loss, base_loss_logs = torch.tensor([0.0]), {}

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            try:
                d_weight = self.calculate_adaptive_weight(base_loss, g_loss, last_layer=last_layer)
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = base_loss + d_weight * disc_factor * g_loss

            log = base_loss_logs
            log.update({
                "{}/total_loss".format(split): loss.clone().detach().mean(),
                "{}/base_loss".format(split): base_loss.detach().mean(),
                "{}/d_weight".format(split): d_weight.detach(),
                "{}/disc_factor".format(split): torch.tensor(disc_factor),
                "{}/g_loss".format(split): g_loss.detach().mean(),
            })
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {
                "{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                "{}/logits_real".format(split): logits_real.detach().mean(),
                "{}/logits_fake".format(split): logits_fake.detach().mean()
            }
            return d_loss, log
