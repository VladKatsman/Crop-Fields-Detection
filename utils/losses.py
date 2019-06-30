import torch
from torch import nn


class ComboLoss(nn.Module):
    def __init__(self, weights, channel_weights=[1, 0.5, 0.5], channel_losses=None):
        super().__init__()
        self.weights = weights
        self.dice = DiceLoss(per_image=False)
        self.focal = FocalLoss2d()
        self.mapping = {'dice': self.dice,
                        'focal': self.focal}
        self.expect_sigmoid = {'dice', 'focal'}
        self.per_channel = {'dice'}
        self.values = {}
        self.channel_weights = channel_weights
        self.channel_losses = channel_losses

    def forward(self, outputs, targets):
        loss = 0
        weights = self.weights
        sigmoid_input = torch.sigmoid(outputs)
        for k, v in weights.items():
            if not v:
                continue
            val = 0
            if k in self.per_channel:
                channels = outputs.size(1)
                for c in range(channels):
                    if not self.channel_losses or k in self.channel_losses[c]:
                        t = torch.eq(targets, c)

                        val += self.channel_weights[c] * self.mapping[k](
                            sigmoid_input[:, c, ...] if k in self.expect_sigmoid else outputs[:, c, ...],
                            t)

            else:
                val = self.mapping[k](sigmoid_input if k in self.expect_sigmoid else outputs, targets)

            self.values[k] = val
            loss += self.weights[k] * val
        return loss.clamp(min=1e-5)


class FocalLoss2d(nn.Module):
    def __init__(self, gamma=2, ignore_index=255):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, outputs, targets):
        outputs = outputs.contiguous()
        one_hot = torch.zeros_like(outputs)
        for c in range(one_hot.shape[1]):
            one_hot[:, c, :, :] = torch.eq(targets, c)
        one_hot = one_hot.contiguous()
        eps = 1e-8
        non_ignored = one_hot.view(-1) != self.ignore_index
        one_hot = one_hot.view(-1)[non_ignored].float()
        outputs = outputs.contiguous().view(-1)[non_ignored]
        outputs = torch.clamp(outputs, eps, 1. - eps)
        one_hot = torch.clamp(one_hot, eps, 1. - eps)
        pt = (1 - one_hot) * (1 - outputs) + one_hot * outputs
        return (-(1. - pt) ** self.gamma * torch.log(pt)).mean()


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, per_image=False):
        super().__init__()
        self.size_average = size_average
        self.register_buffer('weight', weight)
        self.per_image = per_image

    def forward(self, input, target):
        return soft_dice_loss(input, target, per_image=self.per_image)


def soft_dice_loss(outputs, targets, per_image=False):
    batch_size = outputs.size()[0]
    eps = 1e-5
    if not per_image:
        batch_size = 1
    dice_target = targets.contiguous().view(batch_size, -1).float()
    dice_output = outputs.contiguous().view(batch_size, -1)
    intersection = torch.sum(dice_output * dice_target, dim=1)
    union = torch.sum(dice_output, dim=1) + torch.sum(dice_target, dim=1) + eps
    loss = (1 - (2 * intersection + eps) / union).mean()

    return loss
