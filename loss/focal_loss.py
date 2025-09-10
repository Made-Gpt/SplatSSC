# Copyright (c) OpenMMLab. All rights reserved.
import torch
import numpy as np
import torch.nn.functional as F

from .base_loss import BaseLoss
from . import GPD_LOSS
from mmcv.ops import sigmoid_focal_loss as _sigmoid_focal_loss


def CE_wo_softmax(pred, target, class_weights=None, ignore_index=None):
    pred = torch.clamp(pred, 1e-6, 1. -1e-6)
    if ignore_index is not None:
        loss = F.nll_loss(torch.log(pred), target, class_weights, ignore_index=ignore_index)
    else:
        loss = F.nll_loss(torch.log(pred), target, class_weights)
    return loss


@GPD_LOSS.register_module()
class FocalLoss(BaseLoss):

    def __init__(self, weight=1.0, gamma=2.0, alpha=None, ignore_label=0, empty_idx=12,
                 cls_freq=None, median_weight=False, input_dict=None, ignore_empty=False,
                 use_softmax=True, use_label_map=True, use_custom=False, base_loss="nll",
                 **kwargs):  # use_softmax=False
        """
        `Focal Loss <https://arxiv.org/abs/1708.02002>`_
        Args:
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float, optional): A balanced form for Focal Loss.
                Defaults to 0.25.
        """
        super().__init__(weight)

        if input_dict is None:
            self.input_dict = {
                'pred': 'ce_input',
                'label': 'ce_label'
            }
        else:
            self.input_dict = input_dict
        self.loss_func = self.focal_loss
        self.gamma = gamma
        self.alpha = alpha
        self.empty_idx = empty_idx
        self.ignore_label = ignore_label

        # larger the cls_freq, smaller the cls_weight
        if median_weight:
            cls_freq_array = np.array(cls_freq, dtype=np.float32)
            median_freq = np.median(cls_freq_array)
            weights = median_freq / (cls_freq_array + 1e-6)
            np.clip(weights, a_min=None, a_max=15.0, out=weights)
            self.cls_weight = torch.from_numpy(weights).cuda()
        else:
            self.cls_weight = torch.from_numpy(1 / np.log(cls_freq)).cuda()
        self.ignore_empty = ignore_empty
        self.use_label_map = use_label_map
        self.use_softmax = use_softmax
        self.use_custom = use_custom
        self.base_loss = base_loss

        H, W = 256, 256  # hard coding
        xy, yx = torch.meshgrid([torch.arange(H)-H/2,  torch.arange(W)-W/2])
        c = torch.stack([xy,yx], 2)
        c = torch.norm(c, 2, -1)
        c_max = c.max()
        self.c = (c/c_max + 1).cuda()

    def focal_loss(self, pred, label, fov_mask):
        bs, num_cls = pred.shape[:2]

        pred = pred.permute(0, 2, 3, 4, 1).reshape(bs, -1, num_cls)  # [bs, N, 12]
        target_cache = label.long().reshape(bs, -1)  # [bs, N]
        fov_mask = fov_mask.reshape(bs, -1)  # [bs, N]

        target = target_cache.clone()
        if self.use_label_map:
            loc_ignore = 255
            loc_empty = 0
            target[target_cache == self.ignore_label] = loc_ignore
            target[target_cache == self.empty_idx] = loc_empty
        else:
            loc_ignore = self.ignore_label
            loc_empty = self.empty_idx

        non_ignore_mask = (target != loc_ignore)  # [bs, N]

        if self.ignore_empty:
            non_empty_mask = (target != loc_empty)  # [bs, N]
            final_mask = non_ignore_mask & fov_mask & non_empty_mask  # [bs, N]
        else:
            final_mask = non_ignore_mask & fov_mask  # [bs, N]

        pred_masked = pred[final_mask]  # Shape: [num_valid_voxels_in_batch, 12]
        target_masked = target[final_mask]  # Shape: [num_valid_voxels_in_batch]

        if self.use_softmax:
            c = torch.ones_like(target).reshape(-1).cuda() # 129600
            visible_mask = final_mask.reshape(-1).nonzero().squeeze(-1)
            weight_mask = self.cls_weight[None, :] * c[visible_mask, None]

            loss_cls = self.sigmoid_focal_loss(pred_masked, target_masked, weight_mask)
        else:
            class_weights = self.cls_weight.to(dtype=pred.dtype)
            if self.use_custom:
                loss_cls = self.custom_focal_loss(pred_masked, target_masked, class_weights=class_weights, ignore_index=loc_ignore)
            else:
                loss_cls = CE_wo_softmax(pred_masked, target_masked, class_weights=class_weights, ignore_index=loc_ignore)

        return loss_cls

    def sigmoid_focal_loss(self, pred, target, weight=None):

        r"""A wrapper of cuda version `Focal Loss
        <https://arxiv.org/abs/1708.02002>`_.
        """
        loss = _sigmoid_focal_loss(pred.contiguous(), target.contiguous(), self.gamma, self.alpha, None, 'none')
        if weight is not None:
            if weight.shape != loss.shape:
                if weight.size(0) == loss.size(0):
                    weight = weight.view(-1, 1)
                else:
                    assert weight.numel() == loss.numel()
                    weight = weight.view(loss.size(0), -1)
            assert weight.ndim == loss.ndim
            loss = loss * weight
        loss = loss.sum(-1).mean()
        return loss

    def custom_focal_loss(self, pred, target, class_weights=None, empty_idx=0, ignore_index=255):
        pred = torch.clamp(pred, 1e-6, 1. - 1e-6)
        if self.base_loss == 'nll':
            # for probability
            ce_loss = F.nll_loss(torch.log(pred), target, class_weights, ignore_index=ignore_index, reduction='none')
            p_t = torch.gather(pred, 1, target.unsqueeze(1)).squeeze()
        elif self.base_loss == 'ce':
            # for logit_s
            ce_loss = F.cross_entropy(pred, target, weight=class_weights, ignore_index=ignore_index, reduction='none')
            p_t = torch.exp(-ce_loss)
        else:
            raise ValueError(f"'base_loss' should be either 'ce' or 'nll' !")

        # gamma modulate
        modulating_factor = (1 - p_t) ** self.gamma

        # alpha modulate
        if self.alpha is not None:
            alpha_weights = torch.full_like(target, fill_value=(1.0 - self.alpha), dtype=torch.float32)
            alpha_weights[target == empty_idx] = self.alpha  # negative class
        else:
            alpha_weights = 1.0

        # modulated focal loss
        focal_loss = alpha_weights * modulating_factor * ce_loss
        valid_mask = target != ignore_index

        return focal_loss[valid_mask].mean()

"""

    def focal_loss(self, pred, label, fov_mask):
        pred = pred.float()  # 0 (empty), 1~11 (valid), [bs, 12, 60, 60, 36]
        target_cache = label.long()  # 0 (ignore), 1~11 (valid), 12 (empty), [bs, 60, 60, 36]
        target = target_cache.clone()

        # map label first
        if self.use_label_map:
            loc_ignore=255
            loc_empty = 0
            target[target_cache == self.ignore_label] = loc_ignore  # ignore (0) --> 255
            target[target_cache == self.empty_idx] = loc_empty  # empty (12) --> 0
        else:
            loc_ignore = self.ignore_label
            loc_empty = self.empty_idx
        c = torch.ones_like(target).reshape(-1).cuda()  # bs * 129600

        non_ignore_mask = (target != loc_ignore).squeeze(0)
        if self.ignore_empty:
            non_empty_mask = (target != loc_empty).squeeze(0)
            visible_mask = (non_ignore_mask & fov_mask & non_empty_mask).reshape(-1).nonzero().squeeze(-1)
        else:
            visible_mask = (non_ignore_mask & fov_mask).reshape(-1).nonzero().squeeze(-1)

        # valid weight
        weight = self.cls_weight[None,:] * c[visible_mask, None]

        num_classes = pred.size(1)
        pred = pred.permute(0, 2, 3, 4, 1).reshape(-1, num_classes)[visible_mask]
        target = target.reshape(-1)[visible_mask]

        if self.use_softmax:
            loss_cls = self.sigmoid_focal_loss(pred, target, weight)
        else:
            class_weights = self.cls_weight.to(dtype=pred.dtype)
            if self.use_custom:
                loss_cls = self.custom_focal_loss(pred, target, class_weights=class_weights, ignore_index=loc_ignore)
            else:
                loss_cls = CE_wo_softmax(pred, target, class_weights=class_weights, ignore_index=loc_ignore)
        return loss_cls

"""
