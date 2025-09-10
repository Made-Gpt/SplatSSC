from .base_loss import BaseLoss
from . import GPD_LOSS
from utils.lovasz_losses import lovasz_softmax, lovasz_hinge, global_lovasz_softmax
import torch


@GPD_LOSS.register_module()
class LovaszLoss(BaseLoss):

    def __init__(self, weight=1.0, ignore_label=None, empty_idx=12,
                 use_custom=False, use_softmax=True, use_label_map=True,
                 input_dict=None, **kwargs):  # use_softmax = False
        super().__init__(weight)

        if input_dict is None:
            self.input_dict = {
                'lovasz_input': 'lovasz_input',
                'lovasz_label': 'lovasz_label'
            }
        else:
            self.input_dict = input_dict
        self.loss_func = self.lovasz_loss
        self.ignore_label = ignore_label
        self.empty_idx = empty_idx

        self.use_custom = use_custom
        self.use_softmax = use_softmax
        self.use_label_map = use_label_map

    def lovasz_loss(self, lovasz_input, lovasz_label, fov_mask):
        # input: bs, c, h, w, z
        # output: bs, h, w, z
        if self.use_softmax:
            lovasz_input = torch.softmax(lovasz_input.float(), dim=1)
        lovasz_pred = lovasz_input.float()
        lovasz_target = lovasz_label.long().clone()

        if self.use_label_map:
            loc_ignore = 255
            loc_empty = 0
            lovasz_target[lovasz_label == self.ignore_label] = loc_ignore  # ignore (0) --> 255
            lovasz_target[lovasz_label == self.empty_idx] = loc_empty  # empty (12) --> 0
        else:
            loc_ignore = self.ignore_label
        lovasz_loss = lovasz_softmax(lovasz_pred, lovasz_target, use_custom=self.use_custom, ignore=loc_ignore, fov_mask=fov_mask)
        return lovasz_loss

@GPD_LOSS.register_module()
class GlobalLovaszLoss(BaseLoss):

    def __init__(self, weight=1.0, ignore_label=None, input_dict=None, use_softmax=True, **kwargs):  # use_softmax=False
        super().__init__(weight)

        if input_dict is None:
            self.input_dict = {
                'lovasz_input': 'lovasz_input',
                'lovasz_label': 'lovasz_label'
            }
        else:
            self.input_dict = input_dict
        self.loss_func = self.lovasz_loss
        self.ignore_label = ignore_label
        self.use_softmax = use_softmax

    def lovasz_loss(self, lovasz_input, lovasz_label):
        # input: -1, c, h, w, z
        # output: -1, h, w, z
        if self.use_softmax:
            lovasz_input = torch.softmax(lovasz_input.float(), dim=1)
        lovasz_pred = lovasz_input.float()
        lovasz_target = lovasz_label.long()
        lovasz_loss = global_lovasz_softmax(lovasz_pred, lovasz_target, ignore=self.ignore_label)
        return lovasz_loss


@GPD_LOSS.register_module()
class LovaszHingeLoss(BaseLoss):
    
    def __init__(self, weight=1.0, input_dict=None, **kwargs):
        super().__init__(weight)
        
        if input_dict is None:
            self.input_dict = {
                'lovasz_input': 'lovasz_input',
                'lovasz_label': 'lovasz_label'
            }
        else:
            self.input_dict = input_dict
        self.loss_func = self.lovasz_loss
    
    def lovasz_loss(self, lovasz_input, lovasz_label):
        # input: -1, h, w, z
        # output: -1, h, w, z
        lovasz_input = lovasz_input.float()
        lovasz_label = lovasz_label.long()
        lovasz_loss = lovasz_hinge(lovasz_input, lovasz_label)
        return lovasz_loss