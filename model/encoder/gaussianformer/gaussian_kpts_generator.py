# Copyright (c) Horizon Robotics. All rights reserved.
from typing import List, Optional
import torch, numpy as np
import torch.nn as nn
from mmengine import MODELS
from mmengine.model import xavier_init, constant_init, Sequential, BaseModule
from mmcv.cnn import Linear
from .utils import linear_relu_ln, get_rotation_matrix, safe_sigmoid, safe_norm, cartesian
from ...segmentor.gaussian_segmentor.utils import ConvertData
from .utils import get_rotation_matrix, SIGMOID_MAX

@MODELS.register_module()
class SparseGaussian3DKeyPointsGenerator(BaseModule):
    def __init__(
        self,
        save_da='',
        norm_map=False,
        embed_dims=256,
        num_learnable_pts=0,
        fix_scale=None,
        pc_range=None,
        scale_range=None,
        adaptive=True,
    ):
        super(SparseGaussian3DKeyPointsGenerator, self).__init__()
        self.save_da = save_da
        self.norm_map = norm_map
        self.embed_dims = embed_dims
        self.num_learnable_pts = num_learnable_pts
        if fix_scale is None:
            fix_scale = ((0.0, 0.0, 0.0),)
        self.fix_scale = np.array(fix_scale)
        self.num_pts = len(self.fix_scale) + num_learnable_pts # 7
        if num_learnable_pts > 0:
            self.learnable_fc = Linear(self.embed_dims, num_learnable_pts * 3)

        self.adaptive = adaptive
        self.pc_range = pc_range
        self.scale_range = scale_range

    def init_weight(self):
        if self.num_learnable_pts > 0:
            xavier_init(self.learnable_fc, distribution="uniform", bias=0.0)

    def forward(
        self,
        anchor,
        instance_feature=None,
        metas=None,
        curr_name='',
        cam_pc_range=None,
    ):
        nyu_vox_range = metas['nyu_vox_range'].to(anchor.device)
        bs, num_anchor = anchor.shape[:2]
        # convert_data = ConvertData(
        #     curr_name,
        #     self.save_da,
        # )
        xyz = cartesian(anchor[..., :3], nyu_vox_range, norm_map=self.norm_map)  # camera range
        if self.adaptive:
            z_range = xyz[..., 2]
            xyz_near = z_range.min()
            adaptive_ratio = z_range / (xyz_near + 1e-6)
            adaptive_ratio = torch.clamp(adaptive_ratio, min=1.0, max=10.0)  # [1, 21600]
            adaptive_ratio = adaptive_ratio.unsqueeze(-1).unsqueeze(-1)  # [1, 21600, 1, 1]
            # print(f"z range: {xyz_near.item():.5f} - {xyz_far.item():.5f}, "
            #       f"ratio: {(xyz_far / xyz_near).item():.5f}")
        else:
            adaptive_ratio = 1.0
        # convert_data.convert_da_pts_nocolor(xyz.squeeze(0), 'z_cam')

        fix_scale = anchor.new_tensor(self.fix_scale)  # 7, 3
        scale = fix_scale[None, None].tile([bs, num_anchor, 1, 1]) # [1, 21600, 7, 3]
        if self.num_learnable_pts > 0 and instance_feature is not None:
            learnable_scale = (
                safe_sigmoid(self.learnable_fc(instance_feature)
                .reshape(bs, num_anchor, self.num_learnable_pts, 3))
                - 0.5
            )
            scale = torch.cat([scale, learnable_scale], dim=-2)

        gs_scales = safe_sigmoid(anchor[..., None, 3:6]) # [1, 21600, 1, 3]
        gs_scales = self.scale_range[0] + (self.scale_range[1] - self.scale_range[0]) * gs_scales

        key_points = scale * gs_scales * adaptive_ratio  # [1, 21600, 7, 3]
        rots = anchor[..., 6:10]
        rotation_mat = get_rotation_matrix(rots).transpose(-1, -2) # [1, 21600, 3, 3]

        key_points = torch.matmul(
            rotation_mat[:, :, None], key_points[..., None]
        ).squeeze(-1) # [1, 21600, 7, 3]

        key_points = key_points + xyz.unsqueeze(2)
        # key_points_flatten = key_points.flatten(1, 2)
        # convert_data.convert_da_pts_nocolor(key_points_flatten.squeeze(0), 'kpts')

        return key_points


def create_scale_matrix(gs_scales):
    """Convert scale vector to diagonal scale matrix"""
    batch_shape = gs_scales.shape[:-1]
    scale_matrix = torch.zeros(*batch_shape, 3, 3, device=gs_scales.device)
    for i in range(3):
        scale_matrix[..., i, i] = gs_scales[..., i]
    return scale_matrix


def calculate_covariance(rotation_matrix, scale_matrix):
    # S·S^T calculation
    s_st = torch.matmul(scale_matrix, scale_matrix.transpose(-1, -2))
    # R·(S·S^T)·R^T calculation
    rs = torch.matmul(rotation_matrix, s_st)
    covariance = torch.matmul(rs, rotation_matrix.transpose(-1, -2))
    return covariance


def compute_adaptive_scales(covariance, min_scale=0.1, max_scale=0.8):
    """
    Compute adaptive scales based on covariance eigenvalues, ensuring
    larger scales where covariance is small and smaller scales where covariance is large.
    """
    # Get eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)

    # Ensure eigenvalues are positive
    eigenvalues = torch.clamp(eigenvalues, min=1e-6)

    # Normalize eigenvalues to [0, 1] range
    max_eval = torch.max(eigenvalues, dim=-1, keepdim=True)[0]
    eigenvalues_norm = eigenvalues / (max_eval + 1e-6)

    # Apply inverse relationship: SMALL eigenvalue -> LARGE scale
    adaptive_scales_principal = min_scale + (max_scale - min_scale) * (1 - eigenvalues_norm)

    # Create a diagonal matrix with adaptive scales in principal directions
    adaptive_diag = torch.diag_embed(adaptive_scales_principal)

    # Create new covariance matrix based on adaptive scales (S_adaptive^2 in principal directions)
    adapted_cov_principal = adaptive_diag * adaptive_diag

    # Transform back to original coordinate system: U * S_adaptive^2 * U^T
    adapted_cov = torch.matmul(
        torch.matmul(eigenvectors, adapted_cov_principal),
        eigenvectors.transpose(-1, -2)
    )

    return adapted_cov, adaptive_scales_principal, eigenvectors


def offset_kpts(scale_range, offsets_sigmoid_prob, ref_anchor, scale_mode="cov"):
    """
        offsets: [1, num_anchor, num_cam(1), num_ref_pts(3), num_groups(3), num_off_pts(3), 3]
        scale_mode: 'cov' or 'inv_cov'
    """

    gs_scales = safe_sigmoid(ref_anchor[..., None, None, None, 3:6])  # [1, 21600, 1, 1, 1, 1, 3]
    gs_scales = scale_range[0] + (scale_range[1] - scale_range[0]) * gs_scales

    # Original operations from your code
    # offsets_scales = offsets_sigmoid_prob * gs_scales  # [1, num_anchor, num_cam(1), num_ref_pts(3), num_groups(3), num_off_pts(3), 3]
    rots = ref_anchor[..., 6:10]
    rotation_mat = get_rotation_matrix(rots)[:, :, None, None, None]  # [1, num, 1, 1, 1, 1, 3, 3]

    # Create proper scale matrix
    scale_mat = create_scale_matrix(gs_scales.view(*gs_scales.shape[:-1], 3))  # [1, num, 1, 1, 1, 1, 3, 3]

    # Calculate original covariance matrix
    original_cov = calculate_covariance(rotation_mat, scale_mat)  # [1, num, 1, 1, 1, 1, 3, 3]

    # Compute adaptive covariance based on original covariance
    adapted_cov, adaptive_scales_principal, eigenvectors = compute_adaptive_scales(
        original_cov,
        min_scale=scale_range[0],
        max_scale=scale_range[1]
    )  # [1, num, 1, 1, 1, 1, 3, 3]

    # Reshape offsets for matrix multiplication
    # offsets_reshaped = offsets_scales.view(*offsets_scales.shape[:-1], 3, 1)  # [1, num, 1, 1, 3, 3, 3, 1]
    offsets_reshaped = offsets_sigmoid_prob.view(*offsets_sigmoid_prob.shape[:-1], 3, 1)  # [1, num, 1, 1, 3, 3, 3, 1]

    # Transform offsets to principal coordinate system
    offsets_principal = torch.matmul(
        eigenvectors.transpose(-1, -2),
        offsets_reshaped
    )

    # Scale the offsets in principal directions
    scaled_offsets_principal = offsets_principal * adaptive_scales_principal.unsqueeze(-1)

    # Transform back to original coordinate system
    adaptive_scaled_offsets = torch.matmul(
        eigenvectors,
        scaled_offsets_principal
    )

    # Reshape back to original shape
    offsets_kpts = adaptive_scaled_offsets.squeeze(-1)

    return offsets_kpts


