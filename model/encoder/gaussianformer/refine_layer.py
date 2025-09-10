from mmengine.registry import MODELS
from mmengine.model import BaseModule
from mmcv.cnn import Linear, Scale

import torch, torch.nn as nn
import torch.nn.functional as F
from .utils import (
    linear_relu_ln, safe_norm, safe_sigmoid, safe_tanh, safe_inverse_norm, safe_inverse_tanh, safe_inverse_sigmoid,
    SIGMOID_MAX, get_rotation_matrix, GaussianPrediction, cartesian)
from ...segmentor.gaussian_segmentor.utils import ConvertData


xyz_str = ['x', 'y', 'z']


@MODELS.register_module()
class SparseGaussian3DRefinementModule(BaseModule):

    def __init__(
        self,
        save_da=None,
        norm_map=False,  # true--norm, false--sigmoid
        embed_dims=256,
        feat_dims=64,
        pc_range=None,
        scale_range=None,
        simple_mlp=False,
        share_split=False,  # True
        restrict_xyz=False,  # True
        restrict_scale=False,  # True
        refine_rot=True,  # False
        refine_scale=True,  # False
        activate_scale='normalize',  # tanh or normalize
        unit_xyz=None,  # [4.0, 4.0, 1.0]
        semantic_dim=0,  # 13
        include_opa=True,
        include_v=False
    ):
        super(SparseGaussian3DRefinementModule, self).__init__()
        self.save_da = save_da
        self.norm_map = norm_map
        self.embed_dims = embed_dims
        self.feat_dims = feat_dims
        self.output_dim = 10 + int(include_opa) + semantic_dim + int(include_v) * 2
        self.semantic_start = 10 + int(include_opa)
        self.semantic_dim = semantic_dim
        self.include_opa = include_opa
        self.pc_range = pc_range
        self.scale_range = scale_range
        self.refine_scale = refine_scale
        self.refine_rot = refine_rot
        self.restrict_xyz = restrict_xyz
        self.restrict_scale = restrict_scale
        self.activate_scale = activate_scale
        self.unit_xyz = unit_xyz  # [0.1, 0.1, 0.06]

        self.share_split = share_split
        if simple_mlp:
            if self.share_split:
                self.layers = nn.Sequential(
                    Linear(self.embed_dims, self.feat_dims),  # 96 --> 64
                    nn.ReLU(inplace=True),
                    Linear(self.feat_dims, self.feat_dims),  # 64 --> 64
                    nn.ReLU(inplace=True),
                    nn.LayerNorm(feat_dims),
                )
                self.xyz_head = nn.Linear(self.feat_dims, 3)
                # self.scale_head = nn.Linear(self.feat_dims, 3)
                # self.rot_head = nn.Linear(self.feat_dims, 4)
                # self.opa_head = nn.Linear(self.feat_dims, int(include_opa))
                self.geo_head = nn.Linear(self.feat_dims, 3 + 4 + int(include_opa))
                self.sem_head = nn.Linear(self.feat_dims, semantic_dim)
            else:
                self.layers = nn.Sequential(
                    Linear(self.embed_dims, self.feat_dims),  # 96 --> 64
                    nn.LayerNorm(feat_dims),
                    nn.ReLU(inplace=True),
                    Linear(self.feat_dims, self.output_dim),  # 64 --> 22
                )
        else:
            self.layers = nn.Sequential(
                *linear_relu_ln(embed_dims, 2, 2),
                Linear(self.embed_dims, self.output_dim),
                # Scale(1.0 * self.output_dim),
            )

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        metas,
    ):
        output = self.layers(instance_feature + anchor_embed)  # 1, num, 22
        nyu_pc_range = metas[0]['nyu_vox_range'].to(output.device)

        if self.share_split:
            delta_xyz = self.xyz_head(output)  # [1, num, 3]
            anchor_sem = self.sem_head(output)  # [1, num, c]
            # delta_scale = self.scale_head(output)  # [1, num, 3]
            # delta_rot = self.rot_head(output)  # [1, num, 4]
            # anchor_opa = self.opa_head(output)  # [1, num, 1]

            geo_out = self.geo_head(output)
            delta_scale = geo_out[..., :3]  # [1, num, 3]
            delta_rot = geo_out[..., 3:7]  # [1, num, 4]
            anchor_opa = geo_out[..., 7: (10 + int(self.include_opa))]  # [1, num, 1]
        else:
            delta_xyz = output[..., :3]  # [1, num, 3]
            delta_scale = output[..., 3:6]  # [1, num, 3]
            delta_rot = output[..., 6:10]  # [1, num, 4]
            anchor_opa = output[..., 10: (10 + int(self.include_opa))]
            anchor_sem = output[..., self.semantic_start: (self.semantic_start + self.semantic_dim)]
        unit_xyz =  torch.tensor(self.unit_xyz, device=output.device)  # [3]

        ''' refine location -- add xyz offsets '''
        if self.restrict_xyz:
            delta_xyz_prob = 2 * safe_sigmoid(delta_xyz) - 1  # -1 ~ 1
            delta_xyz = delta_xyz_prob * unit_xyz[None, None, :]  # [1, num, 3]
            # norm to 0~1
            if self.norm_map:
                anchor_cam = safe_norm(anchor[..., :3], nyu_pc_range)
            else:
                anchor_cam = safe_sigmoid(anchor[..., :3])  # [1, num_anchor, num_ref_pts, 3]
            # delta refine in 0~1
            refined_part_sigmoid = anchor_cam + delta_xyz
            # rescale to original range
            if self.norm_map:
                anchor_xyz = safe_inverse_norm(refined_part_sigmoid, nyu_pc_range)
            else:
                anchor_xyz = safe_inverse_sigmoid(refined_part_sigmoid)
        else:
            anchor_xyz = anchor[..., :3] + delta_xyz
        # output = torch.cat([anchor_xyz, output[..., 3:]], dim=-1)
        # for i in range(delta_xyz.shape[-1]):
        #     print(f"anchor_xyz {xyz_str[i]}: {anchor_xyz[..., i].min():.4f}~{anchor_xyz[..., i].max():.4f}, delta_xyz {xyz_str[i]}: {delta_xyz[..., i].min():.4f}~{delta_xyz[..., i].max():.4f}, ")

        ''' refine gaussian scale '''
        scale_size = self.scale_range[1] - self.scale_range[0]  # 0.08 - 0.01 = 0.07
        if self.refine_scale:
            if self.restrict_scale:
                sigmoid_delta_scale = 2 * safe_sigmoid(delta_scale) - 1  # -1 ~ 1
                update_scale = anchor[..., 3:6] + sigmoid_delta_scale * scale_size
                anchor_scale = torch.clamp(update_scale, self.scale_range[0], self.scale_range[1])  # 0.01 ~ 0.08
            else:
                anchor_scale = anchor[..., 3:6] + delta_scale
        else:
            if self.restrict_scale:
                if self.activate_scale == 'tanh':
                    delta_scale = torch.tanh(delta_scale) * SIGMOID_MAX  # [?, ?] --> [-1, 1] --> [-4.959, 4.959]
                elif self.activate_scale == 'normalize':
                    delta_scale = torch.nn.functional.normalize(delta_scale, dim=-1) * SIGMOID_MAX
            anchor_scale = delta_scale
        # output = torch.cat([output[..., :3], anchor_scale, output[..., 6:]], dim=-1)
        # for i in range(delta_scale.shape[-1]):
        #     print(f"scale_final {xyz_str[i]}: {anchor_scale[..., i].min():.4f}~{anchor_scale[..., i].max():.4f}, delta_scale {xyz_str[i]}: {delta_scale[..., i].min():.4f}~{delta_scale[..., i].max():.4f}, ")
        # print(f"{64 * '-'}")

        ''' refine gaussian rotation -- add quaternion offsets '''
        if self.refine_rot:
            delta_rot_norm = torch.nn.functional.normalize(delta_rot, dim=-1)
            delta_w1, delta_x1, delta_y1, delta_z1 = delta_rot_norm[..., 0], delta_rot_norm[..., 1], delta_rot_norm[..., 2], delta_rot_norm[..., 3]
            w1, x1, y1, z1 = anchor[..., 6], anchor[..., 7], anchor[..., 8], anchor[..., 9]
            w_final = delta_w1 * w1 - delta_x1 * x1 - delta_y1 * y1 - delta_z1 * z1
            x_final = delta_w1 * x1 + delta_x1 * w1 + delta_y1 * z1 - delta_z1 * y1
            y_final = delta_w1 * y1 - delta_x1 * z1 + delta_y1 * w1 + delta_z1 * x1
            z_final = delta_w1 * z1 + delta_x1 * y1 - delta_y1 * x1 + delta_z1 * w1
            w_final = w_final.unsqueeze(-1)
            x_final = x_final.unsqueeze(-1)
            y_final = y_final.unsqueeze(-1)
            z_final = z_final.unsqueeze(-1)
            rot_final = torch.cat([w_final, x_final, y_final, z_final], dim=-1)
            anchor_rot = torch.nn.functional.normalize(rot_final, dim=-1)
        else:
            anchor_rot = torch.nn.functional.normalize(delta_rot, dim=-1)
        # output = torch.cat([output[..., :6], anchor_rot, output[..., 10:]], dim=-1)
        # for i in range(delta_rot.shape[-1]):
        #     print(f"anchor_rot [{i}]: {anchor_rot[..., i].min():.4f} ~ {anchor_rot[..., i].max():.4f}, delta_rot [{i}]: {delta_rot[..., i].min():.4f} ~ {delta_rot[..., i].max():.4f}")

        output = torch.cat([anchor_xyz, anchor_scale, anchor_rot, anchor_opa, anchor_sem], dim=-1)  # 1, num, 22

        return output  # output: [1, num, 22]


