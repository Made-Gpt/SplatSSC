import torch.nn as nn, torch
from typing import NamedTuple
from torch import Tensor
from pathlib import Path

import copy
import numpy as np
import open3d as o3d
import torch.nn.functional as F

from mmengine import MODELS
from mmengine.model import BaseModule, Sequential
from mmcv.cnn import Linear, build_activation_layer, build_norm_layer
from mmcv.cnn.bricks.drop import build_dropout


SIGMOID_MAX = 4.595
TANH_MAX = 2.646
LOGIT_MAX = 0.99

class GaussianPrediction(NamedTuple):
    means: Tensor
    scales: Tensor
    rotations: Tensor
    harmonics: Tensor
    opacities: Tensor
    semantics: Tensor


def linear_relu_ln(embed_dims, in_loops, out_loops, input_dims=None):
    if input_dims is None:
        input_dims = embed_dims
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(Linear(input_dims, embed_dims))
            layers.append(nn.ReLU(inplace=True))
            input_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return layers


def safe_sigmoid(tensor):
    tensor = torch.clamp(tensor, -SIGMOID_MAX, SIGMOID_MAX)
    return torch.sigmoid(tensor)


def safe_tanh(tensor):
    tensor = torch.clamp(tensor, -TANH_MAX, TANH_MAX)
    return torch.tanh(tensor)


def safe_norm(xyz, pc_range):
    """ limit to pc_range: [-5, -6, -3, 5, 6, 8] """
    if not xyz.device == pc_range.device:
        pc_range = pc_range.to(xyz.device)

    if len(pc_range.shape) == 1:
        pc_min = pc_range[:3]
        pc_max = pc_range[3:]
    elif len(pc_range.shape) == 2:
        bs = xyz.shape[0]
        pc_min = pc_range[:, :3].view(bs, 1, 3)
        pc_max = pc_range[:, 3:].view(bs, 1, 3)
    elif len(pc_range.shape) == 3:
        pc_min = pc_range[..., :3]
        pc_max = pc_range[..., 3:]
    else:
        raise ValueError("Shape of pc_range should be like [6] or [bs, 6] or [bs, 1, 6]")

    xyz_clamp = torch.clamp(xyz, min=pc_min, max=pc_max)
    # normalize
    xyz_norm = (xyz_clamp - pc_min) / (pc_max - pc_min)
    return xyz_norm


def safe_inverse_sigmoid(tensor):
    tensor = torch.clamp(tensor, 1 - LOGIT_MAX, LOGIT_MAX)
    return torch.log(tensor / (1 - tensor))


def safe_inverse_norm(xyz, pc_range):
    if not xyz.device == pc_range.device:
        pc_range = pc_range.to(xyz.device)
    xyz_clamp = torch.clamp(xyz, 1-LOGIT_MAX, LOGIT_MAX)

    if len(pc_range.shape) == 1:
        pc_min = pc_range[:3]
        pc_max = pc_range[3:]
    elif len(pc_range.shape) == 2:
        bs = xyz.shape[0]
        pc_min = pc_range[:, :3].view(bs, 1, 3)
        pc_max = pc_range[:, 3:].view(bs, 1, 3)
    elif len(pc_range.shape) == 3:
        pc_min = pc_range[..., :3]
        pc_max = pc_range[..., 3:]
    else:
        raise ValueError("Shape of pc_range should be like [6] or [bs, 6] or [bs, 1, 6]")
    return xyz_clamp * (pc_max - pc_min) + pc_min


def safe_inverse_tanh(tensor):
    tensor = torch.clamp(tensor, 1 - LOGIT_MAX, LOGIT_MAX)
    return 0.5 * torch.log((1 + tensor) / (1 - tensor))


def batch_quaternion_multiply(q_cam2world, q_cam_list):
    """
    Multiply two batches of quaternions:
    - q_cam2world: [B, 4]
    - q_cam_list: [B, N, 4]
    Returns:
    - result: [B, N, 4]
    """
    # Expand q_cam2world to match q_cam_list
    if len(q_cam2world.shape) == 2:
        q1 = q_cam2world.unsqueeze(1)  # [B, 1, 4]
    else:
        q1 = q_cam2world[None, None, :]  # [1, 1, 4]
    q2 = q_cam_list  # [B, N, 4]

    # Extract components
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]  # [B, 1]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]  # [B, N]

    # Perform quaternion multiplication
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    # Stack the result
    return torch.stack((w, x, y, z), dim=-1)  # [B, N, 4]


def get_rotation_matrix(tensor):
    assert tensor.shape[-1] == 4
    tensor = F.normalize(tensor, dim=-1)
    mat1 = torch.zeros(*tensor.shape[:-1], 4, 4, dtype=tensor.dtype, device=tensor.device)
    mat1[..., 0, 0] = tensor[..., 0]
    mat1[..., 0, 1] = - tensor[..., 1]
    mat1[..., 0, 2] = - tensor[..., 2]
    mat1[..., 0, 3] = - tensor[..., 3]
    
    mat1[..., 1, 0] = tensor[..., 1]
    mat1[..., 1, 1] = tensor[..., 0]
    mat1[..., 1, 2] = - tensor[..., 3]
    mat1[..., 1, 3] = tensor[..., 2]

    mat1[..., 2, 0] = tensor[..., 2]
    mat1[..., 2, 1] = tensor[..., 3]
    mat1[..., 2, 2] = tensor[..., 0]
    mat1[..., 2, 3] = - tensor[..., 1]

    mat1[..., 3, 0] = tensor[..., 3]
    mat1[..., 3, 1] = - tensor[..., 2]
    mat1[..., 3, 2] = tensor[..., 1]
    mat1[..., 3, 3] = tensor[..., 0]

    mat2 = torch.zeros(*tensor.shape[:-1], 4, 4, dtype=tensor.dtype, device=tensor.device)
    mat2[..., 0, 0] = tensor[..., 0]
    mat2[..., 0, 1] = - tensor[..., 1]
    mat2[..., 0, 2] = - tensor[..., 2]
    mat2[..., 0, 3] = - tensor[..., 3]
    
    mat2[..., 1, 0] = tensor[..., 1]
    mat2[..., 1, 1] = tensor[..., 0]
    mat2[..., 1, 2] = tensor[..., 3]
    mat2[..., 1, 3] = - tensor[..., 2]

    mat2[..., 2, 0] = tensor[..., 2]
    mat2[..., 2, 1] = - tensor[..., 3]
    mat2[..., 2, 2] = tensor[..., 0]
    mat2[..., 2, 3] = tensor[..., 1]

    mat2[..., 3, 0] = tensor[..., 3]
    mat2[..., 3, 1] = tensor[..., 2]
    mat2[..., 3, 2] = - tensor[..., 1]
    mat2[..., 3, 3] = tensor[..., 0]

    mat2 = torch.conj(mat2).transpose(-1, -2)
    
    mat = torch.matmul(mat1, mat2)
    return mat[..., 1:, 1:]


def safe_get_quaternion(R):
    assert R.shape[-2:] == (3, 3), "Input must be (..., 3, 3)"

    # Compute squared components
    four_w = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    four_x = R[..., 0, 0] - R[..., 1, 1] - R[..., 2, 2]
    four_y = -R[..., 0, 0] + R[..., 1, 1] - R[..., 2, 2]
    four_z = -R[..., 0, 0] - R[..., 1, 1] + R[..., 2, 2]

    # Stack and find max index
    stacked = torch.stack([four_w, four_x, four_y, four_z], dim=-1)  # (..., 4)
    max_indices = torch.argmax(stacked, dim=-1)  # (...,)

    q0 = torch.zeros_like(four_w)
    q1 = torch.zeros_like(four_w)
    q2 = torch.zeros_like(four_w)
    q3 = torch.zeros_like(four_w)

    eps = 1e-8  # To avoid division by zero

    # Compute each case separately using masking
    for idx in range(4):
        mask = (max_indices == idx)
        max_val = stacked[..., idx].masked_select(mask)
        q_max = torch.sqrt(max_val + 1.0) / 2.0
        mult = 1.0 / (4.0 * q_max + eps)

        if idx == 0:
            q0.masked_scatter_(mask, q_max)
            q1.masked_scatter_(mask, ((R[..., 2, 1] - R[..., 1, 2]).masked_select(mask)) * mult)
            q2.masked_scatter_(mask, ((R[..., 0, 2] - R[..., 2, 0]).masked_select(mask)) * mult)
            q3.masked_scatter_(mask, ((R[..., 1, 0] - R[..., 0, 1]).masked_select(mask)) * mult)
        elif idx == 1:
            q0.masked_scatter_(mask, ((R[..., 2, 1] - R[..., 1, 2]).masked_select(mask)) * mult)
            q1.masked_scatter_(mask, q_max)
            q2.masked_scatter_(mask, ((R[..., 1, 0] + R[..., 0, 1]).masked_select(mask)) * mult)
            q3.masked_scatter_(mask, ((R[..., 0, 2] + R[..., 2, 0]).masked_select(mask)) * mult)
        elif idx == 2:
            q0.masked_scatter_(mask, ((R[..., 0, 2] - R[..., 2, 0]).masked_select(mask)) * mult)
            q1.masked_scatter_(mask, ((R[..., 0, 1] + R[..., 1, 0]).masked_select(mask)) * mult)
            q2.masked_scatter_(mask, q_max)
            q3.masked_scatter_(mask, ((R[..., 2, 1] + R[..., 1, 2]).masked_select(mask)) * mult)
        else:
            q0.masked_scatter_(mask, ((R[..., 1, 0] - R[..., 0, 1]).masked_select(mask)) * mult)
            q1.masked_scatter_(mask, ((R[..., 0, 2] + R[..., 2, 0]).masked_select(mask)) * mult)
            q2.masked_scatter_(mask, ((R[..., 2, 1] + R[..., 1, 2]).masked_select(mask)) * mult)
            q3.masked_scatter_(mask, q_max)

    quaternion = torch.stack((q0, q1, q2, q3), dim=-1)
    quaternion = quaternion / torch.norm(quaternion, dim=-1, keepdim=True)
    return quaternion
    

def get_quaternion(R):

    assert R.shape[-2:] == (3, 3), "Input must be a 3x3 matrix"

    q0 = torch.sqrt(torch.max(torch.tensor(0.0, dtype=R.dtype, device=R.device), 1 + R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2])) / 2
    q1 = torch.sqrt(torch.max(torch.tensor(0.0, dtype=R.dtype, device=R.device), 1 + R[..., 0, 0] - R[..., 1, 1] - R[..., 2, 2])) / 2
    q2 = torch.sqrt(torch.max(torch.tensor(0.0, dtype=R.dtype, device=R.device), 1 - R[..., 0, 0] + R[..., 1, 1] - R[..., 2, 2])) / 2
    q3 = torch.sqrt(torch.max(torch.tensor(0.0, dtype=R.dtype, device=R.device), 1 - R[..., 0, 0] - R[..., 1, 1] + R[..., 2, 2])) / 2
    
    # Determine the signs of q1, q2, q3 based on the elements of R
    q1 = torch.copysign(q1, R[..., 2, 1] - R[..., 1, 2])
    q2 = torch.copysign(q2, R[..., 0, 2] - R[..., 2, 0])
    q3 = torch.copysign(q3, R[..., 1, 0] - R[..., 0, 1])
    
    # Stack into a single quaternion tensor
    quaternion = torch.stack((q0, q1, q2, q3), dim=-1)
    
    # Normalize the quaternion to ensure it is unit-length
    quaternion = quaternion / torch.norm(quaternion, dim=-1, keepdim=True)

    return quaternion


def cartesian(anchor, pc_range, norm_map=False):
    if norm_map:
        xyz = safe_norm(anchor[..., :3], pc_range)  # [1, num, 3]
    else:
        xyz = safe_sigmoid(anchor[..., :3])  # [1, num, 3]

    if len(pc_range.shape) != 1:
        bs = anchor.shape[0]
        pc_range = pc_range.view(bs, 1, 6)
        xxx = xyz[..., 0] * (pc_range[..., 3] - pc_range[..., 0]) + pc_range[..., 0]
        yyy = xyz[..., 1] * (pc_range[..., 4] - pc_range[..., 1]) + pc_range[..., 1]
        zzz = xyz[..., 2] * (pc_range[..., 5] - pc_range[..., 2]) + pc_range[..., 2]
    else:
        xxx = xyz[..., 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
        yyy = xyz[..., 1] * (pc_range[4] - pc_range[1]) + pc_range[1]
        zzz = xyz[..., 2] * (pc_range[5] - pc_range[2]) + pc_range[2]

    xyz = torch.stack([xxx, yyy, zzz], dim=-1)

    return xyz  # [1, num, 3]


@MODELS.register_module()
class AsymmetricFFN(BaseModule):
    def __init__(
        self,
        in_channels=None,
        pre_norm=None,
        embed_dims=256,
        feedforward_channels=1024,
        num_fcs=2,
        act_cfg=dict(type="ReLU", inplace=True),
        ffn_drop=0.0,
        dropout_layer=None,
        add_identity=True,
        init_cfg=None,
        **kwargs,
    ):
        super(AsymmetricFFN, self).__init__(init_cfg)
        assert num_fcs >= 2, (
            "num_fcs should be no less " f"than 2. got {num_fcs}."
        )
        self.in_channels = in_channels
        self.pre_norm = pre_norm
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)

        layers = []
        if in_channels is None:
            in_channels = embed_dims
        if pre_norm is not None:
            self.pre_norm = build_norm_layer(pre_norm, in_channels)[1]

        for _ in range(num_fcs - 1):
            layers.append(
                Sequential(
                    Linear(in_channels, feedforward_channels),
                    self.activate,
                    nn.Dropout(ffn_drop),
                )
            )
            in_channels = feedforward_channels
        layers.append(Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = Sequential(*layers)
        self.dropout_layer = (
            build_dropout(dropout_layer)
            if dropout_layer
            else torch.nn.Identity()
        )
        self.add_identity = add_identity
        if self.add_identity:
            self.identity_fc = (
                torch.nn.Identity()
                if in_channels == embed_dims
                else Linear(self.in_channels, embed_dims)
            )

    def forward(self, x, identity=None):
        if self.pre_norm is not None:
            x = self.pre_norm(x)
        out = self.layers(x)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        identity = self.identity_fc(identity)
        return identity + self.dropout_layer(out)  # add
