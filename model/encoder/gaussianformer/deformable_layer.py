# Copyright (c) Horizon Robotics. All rights reserved.
from typing import List, Optional
import torch, numpy as np
import torch.nn as nn
from mmengine import MODELS
from mmengine.model import xavier_init, constant_init, Sequential, BaseModule
from mmcv.cnn import Linear
from .utils import linear_relu_ln, get_rotation_matrix, safe_sigmoid, safe_norm
from ...segmentor.gaussian_segmentor.utils import ConvertData


@MODELS.register_module()
class DeformableFeatureAggregation(BaseModule):
    def __init__(
        self,
        norm_map: bool = False,  # true--norm, false--sigmoid
        save_da: str = '',
        embed_dims: int = 256,
        num_groups: int = 8,
        num_levels: int = 4,
        num_cams: int = 6,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        kps_generator: dict = None,
        use_deformable_func=False,
        use_camera_embed=False,
        residual_mode="add",
    ):
        super(DeformableFeatureAggregation, self).__init__()
        try:
            from model.encoder.gaussianformer.ops.pointops import DeformableAggregationFunction as DAF
            print(f"{'== DAF IMPORTED ==':=^36}\n :: succeeded to import DAF to deformable layer! {'== DAF IMPORTED ==':=^36}\n")
        except:
            DAF = None
            print(f"{'!! DAF IMPORTED !!':!^36}\n :: failed to import DAF to deformable layer... {'!! DAF IMPORTED !!':!^36}\n")
        self.DAF = DAF

        if embed_dims % num_groups != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_groups, "
                f"but got {embed_dims} and {num_groups}"
            )
        self.save_da = save_da
        self.norm_map = norm_map
        self.group_dims = int(embed_dims / num_groups) # 32
        self.embed_dims = embed_dims # 96
        self.num_levels = num_levels # 3
        self.num_groups = num_groups # 3
        self.num_cams = num_cams # 1
        self.use_deformable_func = use_deformable_func and self.DAF is not None
        self.attn_drop = attn_drop # 0.15
        self.residual_mode = residual_mode # "add"
        self.proj_drop = nn.Dropout(proj_drop)
        kps_generator["embed_dims"] = embed_dims
        self.kps_generator = MODELS.build(kps_generator)
        self.num_pts = self.kps_generator.num_pts
        self.output_proj = Linear(embed_dims, embed_dims)

        if use_camera_embed:
            self.camera_encoder = Sequential(
                *linear_relu_ln(embed_dims, 1, 2, 12)
            )
            self.weights_fc = Linear(
                embed_dims, num_groups * num_levels * self.num_pts
            )
        else:
            self.camera_encoder = None
            self.weights_fc = Linear(
                embed_dims, num_groups * num_cams * num_levels * self.num_pts
            )

    def init_weight(self):
        constant_init(self.weights_fc, val=0.0, bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)

    def prepare_metas(self, metas, tensor):
        meta_dict = {}

        # Convert to tensor
        projection_mat = metas[0]['cam2img'].to(tensor.device, tensor.dtype)  # [B, 4, 4]
        if len(projection_mat.shape) == 3:
            projection_mat = projection_mat.unsqueeze(1)  # [B, 1, 4, 4]
        else:
            projection_mat = projection_mat[None, None, :]  # [1, 1, 4, 4]
        bs, N = projection_mat.shape[:2]  # [1, 4]

        # Take first two dims and reverse to [W, H]
        image_wh = metas[0]['img_shape']
        image_wh = torch.tensor(image_wh).to(tensor.device, tensor.dtype)  # [B, H, W]
        if len(image_wh.shape) == 3:
            image_wh = image_wh[..., :2][..., [1, 0]]  # [B * N, 3] → [B * N, 2]
        else:
            image_wh = image_wh[..., :2][..., [1, 0]].unsqueeze(1)  # [B * N, 3] → [B * N, 1, 2]

        meta_dict.update({
            'projection_mat': projection_mat,  # [bs, 1, 4, 4]
            'image_wh': image_wh,  # [bs, 1, 2]
            'vox_origin': metas[0]['vox_origin'].to(tensor.dtype),  # [bs, 3]
            'scene_size': metas[0]['scene_size'].to(tensor.dtype),  # [bs, 3]
            'nyu_vox_range': metas[0]['nyu_vox_range'].to(tensor.dtype),  # [bs, 6]
            }
        )

        return meta_dict

    def forward(
        self,
        instance_feature: torch.Tensor,  # [bs, num, 96]
        anchor: torch.Tensor,  # [bs, num, 22]
        anchor_embed: torch.Tensor,  # [bs, num, 96]
        feature_maps: List[torch.Tensor],
        metas: dict,
        loop_times: int = 0,
    ):
        curr_name = metas[0]['name']
        self.convert_data = ConvertData(curr_name, self.save_da)

        metas = self.prepare_metas(metas, anchor)
        bs, num_anchor = instance_feature.shape[:2]
        key_points = self.kps_generator(anchor, instance_feature, metas, curr_name) # [bs, 21600, 7, 3]

        weights = self._get_weights(
            instance_feature, anchor_embed, metas
        )
        if self.use_deformable_func:
            weights = (
                weights.permute(0, 1, 4, 2, 3, 5)
                .contiguous()
                .reshape(
                    bs,
                    num_anchor * self.num_pts,
                    self.num_cams,
                    self.num_levels,
                    self.num_groups,
                )
            )  # 1, 7 num, 1, 1, 3, 3
            points_2d = (
                self.project_points(
                    key_points,
                    metas["projection_mat"],
                    metas.get("image_wh"),
                )
                .permute(0, 2, 3, 1, 4)
                .reshape(bs, num_anchor * self.num_pts, self.num_cams, 2)
            )  # bs, num1, num2, 2, --> 1, 7*num, 1, 2

            # self.convert_data.convert_occ_image_sampling(points_2d.squeeze(0).squeeze(1), image_rgb, f'ref_{loop_times}')

            temp_features_next = self.DAF.apply(
                *feature_maps, points_2d, weights
            ).reshape(bs, num_anchor, self.num_pts, self.embed_dims)
        else:
            temp_features_next = self.feature_sampling(
                feature_maps,
                key_points,
                metas["projection_mat"],
                metas.get("image_wh"),
            )
            temp_features_next = self.multi_view_level_fusion(
                temp_features_next, weights
            )
        features = temp_features_next # [1, 21600, 7, 96]
        features = features.sum(dim=2)  # fuse multi-point features [1, 21600, 96]
        output = self.proj_drop(self.output_proj(features))
        if self.residual_mode == "add":
            output = output + instance_feature
        elif self.residual_mode == "cat":
            output = torch.cat([output, instance_feature], dim=-1)

        return output # [bs, 21600, 96]

    def _get_weights(self, instance_feature, anchor_embed, metas=None):
        bs, num_anchor = instance_feature.shape[:2]
        feature = instance_feature + anchor_embed # [1, 21600, 96]
        if self.camera_encoder is not None: 
            camera_embed = self.camera_encoder(
                metas["projection_mat"][:, :, :3].reshape(
                    bs, self.num_cams, -1
                )
            )
            feature = feature[:, :, None] + camera_embed[:, None]
        weights = (
            self.weights_fc(feature)
            .reshape(bs, num_anchor, -1, self.num_groups)
            .softmax(dim=-2)
            .reshape(
                bs,
                num_anchor,
                self.num_cams,
                self.num_levels,
                self.num_pts,
                self.num_groups,
            )
        ) # [1, 21600, 1, 3, 7, 3]
        if self.training and self.attn_drop > 0:
            mask = torch.rand(
                bs, num_anchor, self.num_cams, 1, self.num_pts, 1
            )
            mask = mask.to(device=weights.device, dtype=weights.dtype)
            weights = ((mask > self.attn_drop) * weights) / (
                1 - self.attn_drop
            )
        return weights

    @staticmethod
    def project_points(key_points, projection_mat, image_wh=None):
        bs, num_anchor, num_pts = key_points.shape[:3]

        pts_extend = torch.cat(
            [key_points, torch.ones_like(key_points[..., :1])], dim=-1
        ) # [bs, 21600, 7, 4]
        points_2d = torch.matmul(
            projection_mat[:, :, None, None], pts_extend[:, None, ..., None]
        ).squeeze(-1) # [bs, 1, 21600, 7, 4]
        points_2d = points_2d[..., :2] / torch.clamp(
            points_2d[..., 2:3], min=1e-5
        )
        if image_wh is not None:
            points_2d = points_2d / image_wh[:, :, None, None]
        return points_2d

    @staticmethod
    def feature_sampling(
        feature_maps: List[torch.Tensor],
        key_points: torch.Tensor,
        projection_mat: torch.Tensor,
        image_wh: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        num_levels = len(feature_maps)
        num_cams = feature_maps[0].shape[1]
        bs, num_anchor, num_pts = key_points.shape[:3]

        points_2d = DeformableFeatureAggregation.project_points(
            key_points, projection_mat, image_wh
        )

        points_2d = points_2d * 2 - 1 # [bs, 1, 21600, 7, 2]
        points_2d = points_2d.flatten(end_dim=1) # [bs, 21600, 7, 2]

        features = []
        for fm in feature_maps:
            # fm [bs, 1, 96, 28, 36]
            features.append(
                torch.nn.functional.grid_sample(
                    fm.flatten(end_dim=1), points_2d
                )
            )

        features = torch.stack(features, dim=1)
        features = features.reshape(
            bs, num_cams, num_levels, -1, num_anchor, num_pts
        ).permute(
            0, 4, 1, 2, 5, 3
        )  # bs, num_anchor, num_cams, num_levels, num_pts, embed_dims

        return features

    def multi_view_level_fusion(
        self,
        features: torch.Tensor,
        weights: torch.Tensor,
    ):
        bs, num_anchor = weights.shape[:2]
        features = weights[..., None] * features.reshape(
            features.shape[:-1] + (self.num_groups, self.group_dims)
        )
        features = features.sum(dim=2).sum(dim=2)
        features = features.reshape(
            bs, num_anchor, self.num_pts, self.embed_dims
        )
        return features

"""

def prepare_metas(self, metas, tensor):

    meta_dict = {}
    projection_mat = []
    ida_mat = []
    image_wh = []
    for meta in metas:
        # projection_mat.append(meta['lidar2img'])
        # projection_mat.append(meta['world2img'])
        projection_mat.append(meta['cam2img'])
        # ida_mat.append(meta['img_aug_matrix'])
        image_wh.append(meta['img_shape'])

    projection_mat = torch.from_numpy(np.array(
        projection_mat)).to(tensor.device, tensor.dtype)[0]
    # ida_mat = torch.from_numpy(np.array(
    #     ida_mat)).to(tensor.device, tensor.dtype)[0]
    # ida_mat[..., :2, 2] = ida_mat[..., :2, 3]
    # ida_mat[..., :2, 3] = 0.

    # projection_mat = torch.matmul(ida_mat, projection_mat) # matmul([1, 1, 4, 4], [4, 4])
    projection_mat = projection_mat.unsqueeze(0).unsqueeze(0)  # [1, 1, 4, 4]
    bs, N = projection_mat.shape[:2]
    image_wh = torch.from_numpy(np.array(
        image_wh)).to(tensor.device, tensor.dtype) # [224, 288, 3]
    image_wh = image_wh[0].unflatten(0, (bs, N))[..., [1, 0]] # [288., 224.]

    meta_dict.update({
        'projection_mat': projection_mat,
        'image_wh': image_wh,
        'vox_origin': metas[0]['vox_origin'].to(tensor.dtype),
        'scene_size': metas[0]['scene_size'].to(tensor.dtype),
        'nyu_vox_range': metas[0]['nyu_vox_range'].to(tensor.dtype)}
    )
    return meta_dict
        
"""