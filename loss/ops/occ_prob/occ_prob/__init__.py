#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch.nn as nn
import torch
from . import _C


class _OccAggregate(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        pts,
        points_int,
        means3D,
        means3D_int,
        radii,
        cov3D,
        opacities,
        H, W, D
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            pts,
            points_int,
            means3D,
            means3D_int,
            radii,
            cov3D,
            opacities,
            H, W, D
        )
        # Invoke C++/CUDA rasterizer
        num_rendered, bin_logits, geomBuffer, binningBuffer, imgBuffer = _C.occ_aggregate(*args)  # todo

        # Keep relevant tensors for backward
        ctx.num_rendered = num_rendered
        ctx.H = H
        ctx.W = W
        ctx.D = D
        ctx.save_for_backward(
            geomBuffer,
            binningBuffer,
            imgBuffer,
            means3D,
            pts,
            points_int,
            cov3D,
            opacities,
            bin_logits,
        )
        return bin_logits

    @staticmethod  # todo
    def backward(ctx, bin_logits_grad):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        H = ctx.H
        W = ctx.W
        D = ctx.D
        geomBuffer, binningBuffer, imgBuffer, means3D, pts, points_int, cov3D, opacities, bin_logits = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (
            geomBuffer,
            binningBuffer,
            imgBuffer,
            H, W, D,
            num_rendered,
            means3D,
            pts,
            points_int,
            cov3D,
            opacities,
            bin_logits,
            bin_logits_grad)

        # Compute gradients for relevant tensors by invoking backward method
        means3D_grad, opas_grad, cov3D_grad = _C.occ_aggregate_backward(*args)

        grads = (
            None,  # pts,
            None,  # points_int,
            means3D_grad,  # means3D,
            None,  # means3D_int,
            None,  # radii,
            cov3D_grad,  # cov3D,
            opas_grad,  # opacities,
            None, None, None  # H, W, D
        )

        return grads

class OccAggregator(nn.Module):
    def __init__(self, scale_multiplier, H, W, D, pc_min, grid_size, radii_min=1, scale_range=[0.01, 0.16]):
        super().__init__()
        self.scale_multiplier = scale_multiplier
        self.H = H
        self.W = W
        self.D = D

        self.register_buffer('pc_min', torch.tensor(pc_min, dtype=torch.float))
        self.grid_size = grid_size
        self.radii_min = radii_min
        self.scale_range = scale_range

    def forward(
        self,
        pts,
        means3D,
        scales,
        cov3D,
        opacities,
        origin_use=None
    ):

        assert pts.shape[0] == 1
        pts = pts.squeeze(0)
        assert not pts.requires_grad
        means3D = means3D.squeeze(0)
        scales = scales.detach().squeeze(0)
        cov3D = cov3D.squeeze(0)
        opacities = opacities.squeeze(0)

        if origin_use is not None:
            self.pc_min = origin_use

        points_int = ((pts - self.pc_min) / self.grid_size).to(torch.int)
        assert points_int.min() >= 0 and points_int[:, 0].max() < self.H and points_int[:, 1].max() < self.W and points_int[:, 2].max() < self.D
        means3D_int = ((means3D.detach() - self.pc_min) / self.grid_size).to(torch.int)
        assert means3D_int.min() >= 0 and means3D_int[:, 0].max() < self.H and means3D_int[:, 1].max() < self.W and means3D_int[:, 2].max() < self.D

        radii = torch.ceil(scales.max(dim=-1)[0] * self.scale_multiplier / self.grid_size).to(torch.int)
        # radii = torch.ceil(scales.max(dim=-1)[0] * self.scale_multiplier / self.scale_range[1]).to(torch.int)
        radii = radii.clamp(min=self.radii_min)
        assert radii.min() >= 1
        cov3D = cov3D.flatten(1)[:, [0, 4, 8, 1, 5, 2]]

        bin_logits = _OccAggregate.apply(
            pts,
            points_int,
            means3D,
            means3D_int,
            radii,
            cov3D,
            opacities,
            self.H, self.W, self.D
        )

        return bin_logits  # n, c

