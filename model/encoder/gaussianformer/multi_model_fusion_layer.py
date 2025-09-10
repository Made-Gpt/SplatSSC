# Copyright (c) Horizon Robotics. All rights reserved.
import time
from typing import List, Optional
import torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F

from mmengine import MODELS
from mmengine.model import xavier_init, constant_init, Sequential, BaseModule
from mmcv.cnn import Linear
from model.encoder.gaussianformer.utils import linear_relu_ln, get_rotation_matrix, safe_sigmoid, safe_norm
from model.segmentor.gaussian_segmentor.utils import ConvertData

"""
[
    [0, 0, 0],
    [0.45, 0, 0],  # +x
    [-0.225, 0.3897114317027206, 0],  
    [-0.225, -0.3897114317027206, 0],  
    [0, 0, 0.45]  # +z
]
"""

class MLPAttentionPredictor(nn.Module):
    def __init__(self, qk_dim, hidden_dim=96, out_dim=1):
        super(MLPAttentionPredictor, self).__init__()
        self.fc1 = nn.Linear(qk_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        weight = self.fc2(x)
        return weight

@MODELS.register_module()
class MultiModuleFeatureDepthFusion(BaseModule):
    def __init__(
        self,
        norm_map: bool = False,  # true--norm, false--sigmoid
        save_da: str = '',
        q_dims: int = 128,
        kv_dims: int = 96,
        embed_dims: int = 256,
        num_groups: int = 8,
        num_levels: int = 4,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        image_size=(480, 640),
        use_camera_embed=False,
        use_grid_sample=False,
        use_conf_fuse=True,
        residual_mode="add",
        attn_mode="dot_product",  # "mlp"
    ):
        super(MultiModuleFeatureDepthFusion, self).__init__()
        if embed_dims % num_groups != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_groups, " 
                f"but got {embed_dims} and {num_groups}"
            )
        self.save_da = save_da
        self.norm_map = norm_map
        self.q_dims = q_dims
        self.kv_dims = kv_dims
        self.group_dims = int(embed_dims / num_groups)  # 32
        self.embed_dims = embed_dims  # 96
        self.num_levels = num_levels  # 3
        self.num_groups = num_groups  # 3
        self.scale = num_groups ** -0.5
        self.attn_drop = attn_drop  # 0.15
        self.residual_mode = residual_mode  # "add"
        self.use_conf_fuse = use_conf_fuse
        self.use_grid_sample = use_grid_sample
        self.proj_drop = nn.Dropout(proj_drop)
        self.input_proj_q = Linear(q_dims, embed_dims)
        # self.input_proj_kv = Linear(kv_dims, embed_dims*2)
        self.input_proj_k = Linear(kv_dims, embed_dims)
        self.input_proj_v = Linear(kv_dims, embed_dims)
        self.output_proj = Linear(embed_dims, embed_dims)
        self.dropout = nn.Dropout(attn_drop)
        self.offsets = torch.tensor([
            [0, 0],  # low
            [0, 1],  # col++
            [1, 0],  # row++
            [1, 1],  # high
        ])[None, None, :]  # [1, 1, 4, 2]
        self.image_size = image_size  # h, w
        self.attn_mode = attn_mode

        if use_camera_embed:
            self.camera_encoder = Sequential(
                *linear_relu_ln(embed_dims, 1, 2, 12)
            )
        else:
            self.camera_encoder = None

        if self.attn_mode == "mlp":
            # self.weights_fc = MLPAttentionPredictor(self.embed_dims * 5, embed_dims, self.num_groups * self.num_levels)
            self.weights_fc = nn.Linear(self.group_dims * 5, self.num_levels)
        elif self.attn_mode == "add_mlp":
            self.weights_fc = nn.Linear(self.group_dims, 1)
        elif self.attn_mode == 'mlp_mlp':
            self.mm_fuse = nn.Linear(self.group_dims * 2, self.group_dims)
            self.weights_fc = nn.Linear(self.group_dims, 1)

    def init_weight(self):
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)
        xavier_init(self.input_proj_q, distribution="uniform", bias=0.0)
        # xavier_init(self.input_proj_kv, distribution="uniform", bias=0.0)
        xavier_init(self.input_proj_k, distribution="uniform", bias=0.0)
        xavier_init(self.input_proj_v, distribution="uniform", bias=0.0)
        if self.attn_mode != 'dot_product':
            constant_init(self.weights_fc, 0.)
        if self.attn_mode == 'mlp_mlp':
            constant_init(self.mm_fuse, 0.)

    def check_strange(self, value, name, stage):
        assert not torch.isnan(value).any(), f"Warning: {name} contains NaN {stage}!"
        assert not torch.isinf(value).any(), f"Warning: {name} contains inf {stage}!"

    def forward(
        self,
        # anchor: torch.Tensor,  # [bs, num anchor, 3] or [bs, num anchor, 23 or 22]
        sampling_loc: torch.Tensor,  # [bs, num anchor, 2], un-normalized
        conf_map: torch.Tensor,  # [bs, h(392), w(518)]  or [1, num anchor]
        query_feature: torch.Tensor,  # [bs(1), dim(128 or 96), h(392), w(518)], or [bs, num anchor, 96]
        feature_maps: List[torch.Tensor], 
        metas: dict,
    ):
        curr_name = metas[0]['name']
        # cam_k = metas[0]['cam_k'].to(torch.float32)
        self.convert_data = ConvertData(curr_name, self.save_da)
        self.offsets = self.offsets.to(query_feature.device)
        sampling_loc = sampling_loc.to(query_feature.device)

        bs, num_anchor = sampling_loc.shape[:2]

        ''' step1. normalized 2d sampling points '''
        if sampling_loc.shape[-1] == 3:
            # key_pts = sampling_loc  # [bs, num anchor, 3]
            key_pts_2d = sampling_loc  # [bs, num anchor, 3]
        else:
            # key_pts = sampling_loc[..., :3]
            key_pts_2d = sampling_loc[..., :3]

        # key_pts_2d = self.project_points(key_pts, cam_k)  # [bs, num anchor, 2]
        key_pts_2d_extend = key_pts_2d[:, :, None, :]  # [bs, num anchor, 1, 2]

        # normalize to [0, 1]
        hw = torch.tensor(self.image_size, device=key_pts_2d.device)
        key_pts_2d_norm = key_pts_2d_extend / hw  # [bs, num anchor, 1, 2]

        ''' step2. bi-linear sampled multi-scale image features '''
        # [bs, 1, h, w, 96], where [h, w] in [[240, 320], [120, 160], [ 60,  80], [ 30,  40]]
        mc_ms_feat, spatial_shape, scale_start_index = feature_maps
        reference_features, num_scale = self.split_mlvl_img_feats(bs, mc_ms_feat, spatial_shape, scale_start_index)
        # k_values, v_values = self.return_ref_values(reference_features, key_pts_2d_norm)  # [bs, num anchor, 4, 96]
        kv_values = self.return_ref_values(reference_features, key_pts_2d_norm)  # [bs, num anchor, 4, dim]

        # input project k, v
        k_values = self.input_proj_k(kv_values)  # [1, 1, 1200, 4, 96]
        v_values = self.input_proj_v(kv_values)  # [1, 1, 1200, 4, 96]
        # kv_values_map = self.input_proj_kv(kv_values)  # [1, 1, 1200, 4, 192]
        # k_values = kv_values_map[..., :96]  # [1, 1, 1200, 4, 96]
        # v_values = kv_values_map[..., 96:]  # [1, 1, 1200, 4, 96]

        # map confidence map to [0.5, 1], shape: [bs, 1, h, w]
        if conf_map is not None:
            conf_map = torch.sigmoid(torch.clamp((conf_map - 1.0), 0.01, 4.595))
        else:
            self.use_conf_fuse = False

        ''' step3. prepare query feature '''
        # get query value
        if len(query_feature.shape) == 4:  # [bs, 128, h, w] or [bs, 64, h, w]
            # conf_map: [bs, h(392), w(518)]
            q_dim = query_feature.shape[1]
            query_h, query_w = query_feature.shape[2:]
            query_feature = (query_feature
                             .permute(0, 2, 3, 1)
                             .contiguous()
                             .view(bs, query_h, query_w, q_dim)
                             )
            if self.use_conf_fuse:
                # conf_value [bs, num anchor, 1]
                sample_q, sample_c, weight = self.return_query_values(query_feature, key_pts_2d_norm, conf_map)  # [1, 4, num anchor, 96]
                q_value_conf = sample_q * sample_c  # [bs, num anchor, 4, 96]
                value = (q_value_conf * weight).sum(dim = 2)  # [bs, num anchor, 96]
                # identity = value  # [bs, num anchor, 96], residual
                q_value = self.input_proj_q(value)  # [bs, num anchor, 96]
                identity = q_value
            else:
                value = self.return_query_values(query_feature, key_pts_2d_norm)  # [bs, 4, num anchor, 96]
                # identity = value  # [bs, num anchor, 96], residual
                q_value = self.input_proj_q(value)  # [bs, num anchor, 96]
                identity = q_value
        else:  # [1, num anchor, 96]
            if self.use_conf_fuse:
                # conf_map: [bs, num anchor]
                conf_map = conf_map[:, : ,None]  # [bs, num anchor, 1]
                value = query_feature * conf_map  # [bs, num anchor, 96]
            else:
                value = query_feature
            # identity = value  # [bs, num anchor, 96], residual
            q_value = self.input_proj_q(value)  # [bs, num anchor, 96]
            identity = q_value

        # split to multi-heads
        k_values_ = k_values.view(bs, num_anchor, 4, self.num_groups, self.group_dims).permute(0, 3, 2, 4, 1)  # [bs, num heads, 4, 32, num anchor]
        v_values_ = v_values.view(bs, num_anchor, 4, self.num_groups, self.group_dims).permute(0, 3, 2, 1, 4)  # [bs, num heads, 4, num anchor, 32]
        q_value_ = q_value.unsqueeze(2).view(bs, num_anchor, 1, self.num_groups, self.group_dims).permute(0, 3, 2, 1, 4)  # [bs, num heads, 1, num anchor, 32]

        # aggregate
        if self.attn_mode == 'dot_product':
            # attn = torch.matmul(q_value, k_values) * self.scale  # [bs, num heads, 4, num anchor, num anchor]
            attn = torch.einsum('bhlnd, bhldn -> bhlnn', q_value_, k_values_)
            attn = F.softmax(attn, dim=2)  # [1, num heads, 4, num, num], softmax along scale dim
            attn = self.dropout(attn)  # [1, num heads, 4, num, num]
            features = torch.einsum('bhlnn, bhlnd -> bhlnd', attn, v_values_)
        elif self.attn_mode == 'mlp':
            # k_values_ = k_values.reshape(bs, num_anchor, -1)  # [bs, num anchor, 384]
            # qk_values = torch.cat([q_value, k_values_], dim=-1)  # [bs, num anchor, 480]
            # qk_values = torch.cat([q_value, k_values_], dim=-1)  # [bs, num anchor, 480]
            # [bs, num anchor, num group * num scale]
            # attn = (
            #     self.weights_fc(qk_values)  # [bs, num anchor, num scale * num group]
            #     .view(bs, num_anchor, self.num_levels, self.num_groups)  # [bs, num anchor, num scale, num group]
            #     .permute(0, 3, 2, 1)  # [bs, num group, num scale, num_anchor]
            #     .unsqueeze(-1)  # [bs, num group, num scale, num_anchor, 1]
            # )

            k_values_ = k_values.reshape(bs, self.num_groups, -1, num_anchor).permute(0, 1, 3, 2)  # [bs, num group, num anchor, 128]
            q_value_ = q_value_.squeeze(2)  # [bs, num heads, num anchor, 32]
            qk_values = torch.cat([q_value_, k_values_], dim=-1)  # [bs, num group, num anchor, 160]
            attn = (
                self.weights_fc(qk_values)  # [bs, num group, num anchor, num scale]
                .permute(0, 1, 3, 2)  # [bs, num group, num scale, num_anchor]
                .unsqueeze(-1)  # [bs, num group, num scale, num_anchor, 1]
            )
            attn = F.softmax(attn, dim=2)  # [bs, num group, num scale, num_anchor, 1], softmax along scale dim
            features = attn * v_values_ # [bs, num group, num scale, num_anchor, 96]
        elif self.attn_mode == 'add_mlp':
            k_values_ = k_values_.permute(0, 1, 2, 4, 3)
            qk_values = q_value_ + k_values_  # [bs, num heads, 4, num anchor, 32]
            qk_values_flatten = qk_values.reshape(bs, self.num_groups, -1, self.group_dims)  # [bs, num group, num levels * num anchor, 32]
            attn = (
                self.weights_fc(qk_values_flatten)  # [bs, num group, num levels * num anchor, 1]
                .view(bs, self.num_groups, self.num_levels, num_anchor, 1)  # [bs, num group, num scale, num_anchor, 1]
            )
            attn = F.softmax(attn, dim=2)  # [bs, num group, num scale, num_anchor, 1], softmax along scale dim
            features = attn * v_values_ # [bs, num group, num scale, num_anchor, 96]
        elif self.attn_mode == 'mlp_mlp':
            k_values_ = k_values_.permute(0, 1, 2, 4, 3)  # [bs, num heads, 4, num anchor, 32]
            q_value_expanded = q_value_.expand(-1, -1, self.num_levels, -1, -1)  # [bs, num_heads, 4, num_anchor, 32]
            qk_values = torch.cat([q_value_expanded, k_values_], dim=-1)  # [bs, num_heads, 4, num_anchor, 64]
            qk_values_flatten = qk_values.view(bs, self.num_groups, -1, self.group_dims)  # [bs, num group, num scales * num anchor, 32]
            attn = (
                self.weights_fc(qk_values_flatten)  # [bs, num group, num levels * num anchor, 1]
                .view(bs, self.num_groups, self.num_levels, self.num_anchor, 1)
            # [bs, num group, num scale, num_anchor, 1]
            )
            attn = F.softmax(attn, dim=2)  # [bs, num group, num scale, num_anchor, 1], softmax along scale dim
            features = attn * v_values_  # [bs, num group, num scale, num_anchor, 32]
        else:
            raise ValueError(f"unexpected attn_mode {self.attn_mode}")

        # if self.use_conf_fuse:
        #     # add confidence to query dimension
        #     conf_weight = conf_value[None, None, :, :, :]  # [bs, 1, 1, num_anchor, 1]
        #     attn = attn * conf_weight
        # features = torch.matmul(attn, v_values)  # [bs, num heads, 4, num, 32]
        features = features.sum(dim=2)  # [bs, num heads, num, 32]
        features = features.transpose(1,2).reshape(bs, num_anchor, 96)  # fuse multi-scale features [bs, 21600, 96]

        output = self.proj_drop(self.output_proj(features))  # [bs, num anchor, 96]
        if self.residual_mode == "add":
            output = output + identity
        elif self.residual_mode == "cat":
            output = torch.cat([output, identity], dim=-1)

        return output # [bs, 21600, 96]

    def split_mlvl_img_feats(self, bs, mc_ms_feat, spatial_shape, scale_start_index):
        feature_maps = []
        start_index = 0
        num_scale = 0
        for i, hw in enumerate(spatial_shape):
            if i == len(spatial_shape)-1:  # i=3
                end_index = mc_ms_feat.shape[2]  # 10200
            else:
                end_index = scale_start_index[i+1]
            scale_feat = mc_ms_feat[:, :, start_index:end_index, :]  # [bs, 1, loc_h * loc_w, 96]
            scale_feat = scale_feat.reshape(bs, hw[0], hw[1], self.embed_dims)  # [bs, loc_h, loc_w, 96]
            feature_maps.append(scale_feat)
            start_index = end_index
            num_scale += 1
        return feature_maps, num_scale

    @staticmethod
    def project_points(key_pts, cam_k):
        f_l_x, f_l_y = cam_k[0, 0], cam_k[1, 1]
        c_x, c_y = cam_k[0, 2], cam_k[1, 2]
        points_2d_x = f_l_x * key_pts[..., 0] / key_pts[..., 2] + c_x
        points_2d_y = f_l_y * key_pts[..., 1] / key_pts[..., 2] + c_y
        points_2d = torch.stack((points_2d_x, points_2d_y), dim=2)

        return points_2d

    @staticmethod
    def map2grid(spatial_shape, device, norm=True):
        h, w = spatial_shape
        h_size, w_size = 1 / h, 1 / w
        y_centers = torch.arange(h, device=device, dtype=torch.float32) + 0.5  # shape [h]
        x_centers = torch.arange(w, device=device, dtype=torch.float32) + 0.5  # shape [w]
        if norm:
            yv, xv = torch.meshgrid(y_centers/h+h_size/2, x_centers/w+w_size/2, indexing='ij')
        else:
            yv, xv = torch.meshgrid(y_centers, x_centers, indexing='ij')
        grid = torch.stack((yv, xv), dim=-1)  # [h, w, 2]
        return grid

    @staticmethod
    def norm2grid(points, grid_res):
        """
        Args:
            points: [bs, num anchor, 1, 2], normalized points, range is [0, 1]
            grid_res: [2], eg, [480, 640]
        Returns:
            indices of points in target grid, eg, [480, 640]
        """
        # eg, [480, 640]
        if isinstance(grid_res, list):
            grid_res = torch.tensor(grid_res, device=points.device, dtype=torch.float32)

        # eg, 1/480, 1/640
        grid_size_x = 1.0 / grid_res[1]
        grid_size_y = 1.0 / grid_res[0]

        # map normalized points to target grid, and get the floor index
        grid_indices_x = torch.floor(points[..., 0] / grid_size_x)
        grid_indices_y = torch.floor(points[..., 1] / grid_size_y)
        grid_indices = torch.stack([grid_indices_x, grid_indices_y], dim=-1)
        grid_indices[..., 0] = torch.clamp(grid_indices[..., 0], 0.0, grid_res[0] - 1)
        grid_indices[..., 1] = torch.clamp(grid_indices[..., 1], 0.0, grid_res[1] - 1)

        return grid_indices

    def bi_linear_weight(self, key_pts_2d, loc_hw):
        # get floor references
        sample_points = key_pts_2d * loc_hw
        sample_floor = self.norm2grid(key_pts_2d, loc_hw)  # [bs, num anchor, 1, 2]

        # get distances for bi-linear sampling
        low_inter = sample_points - sample_floor  # 0: lh, 1: lw, [bs, num anchor, 1, 2]
        high_inter = 1 - low_inter  # 0: hh, 1: hw, [bs, num anchor, 1, 2]

        # shape: [bs, num anchor, 1, 1], w1, w2, w3, w4 are corresponding to these offsets: [1,1], [1,0], [0,0], [0,1]
        w1, w2 = high_inter[..., 0] * high_inter[..., 1], high_inter[..., 0] * low_inter[..., 1]
        w3, w4 = low_inter[..., 0] * low_inter[..., 1], low_inter[..., 0] * high_inter[..., 1]
        weight = torch.cat([w3, w4, w2, w1], dim=-1).squeeze(2).unsqueeze(-1)  # [bs, num anchor, 4, 1]

        return weight, sample_floor

    def return_ref_values(self, reference_features: list, key_pts_2d: torch.Tensor):
        """
        Args:
            reference_features: 4 scale features, shape of each is  [bs, 1, h, w, 96].
            key_pts_2d: normalized key reference points, [bs, num anchor, 1, 2].
        Returns:
            stacked keys and values, shape of each is [bs, num anchor, 4, 96].
        """
        # k_values, v_values = [], []
        kv_values = []
        # get reference value for different level
        for i, ref_feat in enumerate(reference_features):
            if self.use_grid_sample:
                ref_feat_permute = ref_feat.permute(0, 3, 1, 2)  # [bs, C, h, w]
                grid = key_pts_2d * 2.0 - 1.0  # [bs, num anchor, 1, 2].
                sample_feat = F.grid_sample(
                    input=ref_feat_permute,
                    grid=grid,
                    mode='bilinear',
                    padding_mode='border',  # 'border' corresponds to clamp
                    align_corners=False  # It's recommended to set this to False
                )  # [bs, dim, h * w, 1]
                value = sample_feat.permute(0, 2, 1, 3).squeeze(-1)  # [bs, h * w, dim]
            else:
                b, h, w, d = ref_feat.shape[:]  # [bs, h, w, dim]
                num_anchor = key_pts_2d.shape[1]
                loc_hw = torch.tensor([h, w], device=key_pts_2d.device)
                # weight: [bs, num, 4, 1], sample_floor: [bs, num, 1, 2]
                weight, sample_floor = self.bi_linear_weight(key_pts_2d, loc_hw)

                sample_location = sample_floor + self.offsets  # [bs, num anchor, 4, 2]
                sample_location[..., 0] = torch.clamp(sample_location[..., 0], 0, loc_hw[0]-1)
                sample_location[..., 1] = torch.clamp(sample_location[..., 1], 0, loc_hw[1]-1)
                sample_location = sample_location.squeeze(1).long()  # for index, [bs, num anchor, 4, 2]

                """
                # shape of ref_feat is [B, h, w, dim], sample_feat is corresponding to the following offsets: [0,0], [0,1], [1,0], [1,1]
                # sample_feat = ref_feat[:, sample_location[..., 0], sample_location[..., 1], :].squeeze(1)  # [B, num anchor, 4, dim]
                batch_idx = torch.arange(b, device=ref_feat.device).view(b, 1, 1)
                batch_idx = batch_idx.expand(b, num_anchor, 4)  # Shape: [B, N, 4]
                sample_feat = ref_feat[batch_idx, sample_location[..., 0], sample_location[..., 1]]  # Shape: [B, N, 4, C]
                """
                ref_feat_flat = ref_feat.view(b, h * w, d)  # [B, H*W, D]
                index_flat = sample_location[..., 0] * w + sample_location[..., 1]  # [B, N, 4]
                index_to_gather = index_flat.view(b, num_anchor * 4, 1).expand(-1, -1, d)
                gathered_vals = torch.gather(ref_feat_flat, 1, index_to_gather)
                sample_feat = gathered_vals.view(b, num_anchor, 4, d)  # [B, N, 4, D]

                value = (sample_feat * weight).sum(dim=2)  # [B, num anchor, dim]
                # k_values.append(value[..., :96])
                # v_values.append(value[..., 96:])

            kv_values.append(value)

        kv_values = torch.stack(kv_values, dim=2)  # [B, num anchor, 4, 96]

        return kv_values  # k_values, v_values

    def return_query_values(self, query_feature, key_pts_2d, conf_map=None):
        bs, h, w = query_feature.shape[:3]  # [bs, h, w, 96 or 128]
        loc_hw = torch.tensor([h, w], device=key_pts_2d.device)
        weight, sample_floor = self.bi_linear_weight(key_pts_2d, loc_hw)
        sample_location = sample_floor + self.offsets  # [bs, num anchor, 4, 2]
        sample_location[..., 0] = torch.clamp(sample_location[..., 0], 0, loc_hw[0] - 1)
        sample_location[..., 1] = torch.clamp(sample_location[..., 1], 0, loc_hw[1] - 1)
        sample_location = sample_location.squeeze(1).long()  # for index, [1, num anchor, 4, 96]

        # shape of query feat is [1, h, w, 96 or 128], sample_feat is corresponding to the following offsets: [0,0], [0,1], [1,0], [1,1]
        sample_feat = query_feature[:, sample_location[..., 0], sample_location[..., 1], :].squeeze(1)  # [1, num anchor, 4, 96 or 128]

        if self.use_conf_fuse:
            assert conf_map is not None
            # conf map should have same h, w with query map
            sample_conf = conf_map[:, sample_location[..., 0], sample_location[..., 1]].squeeze(1).unsqueeze(-1)  # [1, num anchor, 4, 1]
            return sample_feat, sample_conf, weight
        else:
            value = (sample_feat * weight).sum(dim=2)  # [B, num anchor, 96]
            return value

        # if self.use_conf_fuse:
        #     assert conf_map is not None
        #     # conf map should have same h, w with query map
        #     sample_conf = conf_map[:, :, sample_location[..., 0], sample_location[..., 1]].squeeze(0).squeeze(1).unsqueeze(-1)  # [1, num anchor, 4, 1]
        #     conf = (sample_conf * weight).sum(dim=2)  # [1, num anchor, 1]
        #     return value, conf
        # else:
        #     return value
