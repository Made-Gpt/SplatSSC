import torch
import torch.nn.functional as F

from torch.cuda.amp import autocast

from .base_loss import BaseLoss
from . import GPD_LOSS

@GPD_LOSS.register_module()
class PCD_Huber_Loss(BaseLoss):
    def __init__(self, weight=1.0, input_dict=None, **kwargs):
        super().__init__(weight)

        if input_dict is None:
            self.input_dict = {
                'point_preds': 'da_pts_input',
                'point_labels': 'da_pts_label',
            }
        else:
            self.input_dict = input_dict
        self.loss_func = self.pcd_huber_loss

    def pcd_huber_loss(self, point_preds, point_labels):
        # point_preds, point_labels: [B, num, 3]
        point_preds = point_preds.float()
        point_labels = point_labels.float()
        loss = F.huber_loss(point_preds, point_labels, reduction='mean', delta=0.1)
        return loss


@GPD_LOSS.register_module()
class Depth_Huber_Loss(BaseLoss):
    def __init__(self, weight=1.0, input_dict=None, **kwargs):
        super().__init__(weight)

        if input_dict is None:
            self.input_dict = {
                'depth_preds': 'da_depth_input',
                'depth_labels': 'da_depth_label'
            }
        else:
            self.input_dict = input_dict
        self.loss_func = self.depth_huber_loss

    def depth_huber_loss(self, depth_preds, depth_labels):
        # pred: [1, h, w], depth_gt: [1, h, w]
        depth_preds = depth_preds.float()
        depth_labels = depth_labels.float()

        mask = (depth_preds != 0) & (depth_labels != 0)
        loss = F.huber_loss(depth_preds[mask], depth_labels[mask], reduction='mean', delta=0.1)
        return loss


@GPD_LOSS.register_module()
class Depth_Gradient_Loss(BaseLoss):
    """
    Computes the L1 loss between the gradients of the predicted depth map
    and the ground truth depth map.
    This loss encourages smooth surfaces in flat regions and sharp edges.
    """

    def __init__(self, weight=1.0, input_dict=None, loss_type='l1', **kwargs):
        super().__init__(weight)

        if input_dict is None:
            self.input_dict = {
                'depth_preds': 'da_depth_input',  # [B, H, W]
                'depth_labels': 'da_depth_label',  # [B, H, W]
                'valid_mask': 'valid_mask'  # [H, W]
            }
        else:
            self.input_dict = input_dict

        self.loss_type = loss_type.lower()
        if self.loss_type not in ['l1', 'huber']:
            raise ValueError(f"Unsupported loss_type for Depth_Gradient_Loss: {loss_type}. Supported: 'l1', 'huber'")

        if self.loss_type == 'huber':
            self.huber_delta = kwargs.get('huber_delta_gradient', 0.1)  # Default delta for Huber loss on gradients

        self.loss_func = self._calculate_depth_gradient_loss

    def _compute_gradients(self, depth_map):
        """ Computes image gradients (dy, dx) for a batch of depth maps. """
        # dy (gradient along height, diff between D[y+1,x] and D[y,x])
        dy = depth_map[:, 1:, :] - depth_map[:, :-1, :]
        # dx (gradient along width, diff between D[y,x+1] and D[y,x])
        dx = depth_map[:, :, 1:] - depth_map[:, :, :-1]

        return dy, dx

    def _calculate_depth_gradient_loss(self, depth_preds, depth_labels, valid_mask):

        depth_preds = depth_preds.float()
        depth_labels = depth_labels.float()
        valid_mask = valid_mask.bool()  # Ensure mask is boolean

        if not (depth_preds.shape == depth_labels.shape == valid_mask.shape):
            raise ValueError(
                f"Shape mismatch: pred {depth_preds.shape}, label {depth_labels.shape}, mask {valid_mask.shape}")

        # 1. Compute gradients
        pred_grad_y, pred_grad_x = self._compute_gradients(depth_preds)
        gt_grad_y, gt_grad_x = self._compute_gradients(depth_labels)

        # 2. Create masks for valid gradients
        mask_grad_y = valid_mask[:, :-1, :] & valid_mask[:, 1:, :]
        mask_grad_x = valid_mask[:, :, :-1] & valid_mask[:, :, 1:]

        loss_grad_x = torch.tensor(0.0, device=depth_preds.device, dtype=depth_preds.dtype)
        loss_grad_y = torch.tensor(0.0, device=depth_preds.device, dtype=depth_preds.dtype)

        # 3. Calculate L1 or Huber loss on gradients where they are valid
        if mask_grad_x.sum() > 0:
            if self.loss_type == 'l1':
                loss_grad_x = F.l1_loss(pred_grad_x[mask_grad_x], gt_grad_x[mask_grad_x], reduction='mean')
            elif self.loss_type == 'huber':
                loss_grad_x = F.huber_loss(pred_grad_x[mask_grad_x], gt_grad_x[mask_grad_x], reduction='mean',
                                           delta=self.huber_delta)

        if mask_grad_y.sum() > 0:
            if self.loss_type == 'l1':
                loss_grad_y = F.l1_loss(pred_grad_y[mask_grad_y], gt_grad_y[mask_grad_y], reduction='mean')
            elif self.loss_type == 'huber':
                loss_grad_y = F.huber_loss(pred_grad_y[mask_grad_y], gt_grad_y[mask_grad_y], reduction='mean',
                                           delta=self.huber_delta)

        gradient_loss = loss_grad_x + loss_grad_y

        return gradient_loss


@GPD_LOSS.register_module()
class Depth_Scale_Loss(BaseLoss):
    def __init__(self, weight=1.0, input_dict=None, **kwargs):
        super().__init__(weight)

        if input_dict is None:
            self.input_dict = {
                'depth_preds': 'da_input',
                'depth_labels': 'da_label'
            }
        else:
            self.input_dict = input_dict
        self.loss_func = self.depth_scal_loss

    def depth_scal_loss(self, depth_preds, depth_labels):
        # pred: [1, h, w], depth_gt: [1, h, w]
        depth_preds = depth_preds.float()
        depth_labels = depth_labels.float()

        mask = (depth_preds != 0) & (depth_labels != 0)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=depth_preds.device)
        pred_valid = depth_preds[mask]
        label_valid = depth_labels[mask]

        eps = 1e-6
        diff = torch.log(pred_valid + eps) - torch.log(label_valid + eps)

        # scale loss：
        term1 = (diff ** 2).mean()
        term2 = (diff.mean()) ** 2

        loss = term1 - term2
        return loss


