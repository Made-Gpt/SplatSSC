import time
import torch
import numpy as np
import open3d as o3d
import torch.nn.functional as F

from torch import nn
from mmengine import MODELS
from model.serialization import CustomSerialization
from typing import List, Optional, Tuple

from ..encoder.gaussianformer.utils import safe_sigmoid
from ..segmentor.gaussian_segmentor.utils import ConvertData, project_points


LOGIT_MAX = 0.99


def safe_inverse_sigmoid(tensor): # 逆 Sigmoid 函数
    tensor = torch.clamp(tensor, 1 - LOGIT_MAX, LOGIT_MAX)
    return torch.log(tensor / (1 - tensor))


def safe_inverse_norm(xyz, pc_range):
    if not xyz.device == pc_range.device:
        pc_range = pc_range.to(xyz.device)
    xyz_clamp = torch.clamp(xyz, 1-LOGIT_MAX, LOGIT_MAX)
    return xyz_clamp * (pc_range[3:] - pc_range[:3]) + pc_range[:3]


def sign_shaft_sigmoid(input):
    input_sign = torch.sign(input)  # [1], 1 or -1
    input_abs = torch.abs(input) * 2  # [1]
    input_adapt = (input_abs - 4.595)  # [1]
    output = safe_sigmoid(input_adapt) * input_sign  # [1], limit to [-1, 1]
    return output


def pcd_voxel_downsample(points, voxel_size=0.1):
    device = points.device
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    if len(points.shape) == 3:
        points = points.reshape(-1, 3)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    return torch.tensor(np.asarray(pcd_down.points), device=device)  # (num, 3)


# def project_points(key_pts, cam_k):
#     # key_pts shape should be [1, num, 3] or [num, 3]
#     f_l_x, f_l_y = cam_k[0, 0], cam_k[1, 1]
#     c_x, c_y = cam_k[0, 2], cam_k[1, 2]
#     points_2d_x = f_l_x * key_pts[..., 0] / key_pts[..., 2] + c_x
#     points_2d_y = f_l_y * key_pts[..., 1] / key_pts[..., 2] + c_y
#     if len(key_pts.shape) == 3:
#         points_2d = torch.stack((points_2d_x, points_2d_y), dim=2)  # [1, num, 2]
#     else:
#         points_2d = torch.stack((points_2d_x, points_2d_y), dim=1)  # [num, 2]
#     return points_2d


# def render_sampled_points_to_image(
#         sampled_points_3d: torch.Tensor,
#         sampled_colors: torch.Tensor,
#         cam_k: torch.Tensor,
#         image_height: int,
#         image_width: int
# ) -> torch.Tensor:
#
#     # --- 1. Points and Colors to Project ---
#     points_to_project = sampled_points_3d
#     colors_to_project = sampled_colors
#
#     # --- 2. Project 3D points to 2D ---
#     uv_proj = project_points(points_to_project, cam_k)
#     u_proj = uv_proj[..., 0]
#     v_proj = uv_proj[..., 1]
#
#     # --- 3. Round 2D coordinates to integer pixel locations ---
#     u_pixels = torch.round(u_proj).long()
#     v_pixels = torch.round(v_proj).long()
#
#     # --- 4. Filter points that project outside the image boundaries ---
#     in_bounds_mask = (u_pixels >= 0) & (u_pixels < image_width) & \
#                      (v_pixels >= 0) & (v_pixels < image_height)
#
#     # If all points project out of bounds, return a black image.
#     # if not torch.any(in_bounds_mask):
#     #     return torch.zeros((image_height, image_width, 3),
#     #                        dtype=sampled_colors.dtype,
#     #                        device=sampled_points_3d.device)
#
#     final_u_coords = u_pixels[in_bounds_mask]
#     final_v_coords = v_pixels[in_bounds_mask]
#     final_colors = colors_to_project[in_bounds_mask]
#
#     # --- 5. Create the output image and "paint" the points ---
#     output_image = torch.zeros((image_height, image_width, 3),
#                                dtype=sampled_colors.dtype,
#                                device=sampled_points_3d.device)
#
#     output_image[final_v_coords, final_u_coords, :] = final_colors
#
#     return output_image


def calculate_structured_downsample_indices(
        orig_h: int,
        orig_w: int,
        out_h: int,
        out_w: int,
        device: torch.device = None
):
    # --- Input Validation (Basic) ---
    if orig_h <= 0 or orig_w <= 0:
        raise ValueError(f"Original dimensions orig_h ({orig_h}) and orig_w ({orig_w}) must be positive.")
    if out_h <= 0 or out_w <= 0:
        raise ValueError(f"Output dimensions out_h ({out_h}) and out_w ({out_w}) must be positive.")

    if device is None:
        device = torch.device('cpu')

    # --- 1. Calculate source indices (mimicking F.interpolate(mode='nearest')) ---
    y_out_coords = torch.arange(out_h, device=device, dtype=torch.float32).view(out_h, 1)  # [out_h, 1]
    x_out_coords = torch.arange(out_w, device=device, dtype=torch.float32).view(1, out_w)  # [1, out_w]

    # Calculate scaling factors (original_dim / output_dim)
    # These determine how many original pixels correspond to one output pixel on average.
    scale_h = float(orig_h) / float(out_h)
    scale_w = float(orig_w) / float(out_w)

    # Compute source row indices (v_orig or r_orig) for each output row, then expand to full map
    src_row_indices = torch.floor(y_out_coords * scale_h).long()
    src_row_indices_map = src_row_indices.expand(-1, out_w)  # Shape: [out_h, out_w]

    # Compute source column indices (u_orig or c_orig) for each output column, then expand to full map
    src_col_indices = torch.floor(x_out_coords * scale_w).long()
    src_col_indices_map = src_col_indices.expand(out_h, -1)  # Shape: [out_h, out_w]

    # Clamp indices to ensure they are within the bounds of the original grid
    src_row_indices_map = torch.clamp(src_row_indices_map, 0, orig_h - 1)
    src_col_indices_map = torch.clamp(src_col_indices_map, 0, orig_w - 1)

    # --- 2. Prepare the (row, col) coordinates in the original grid for the sampled points ---
    # Flatten the 2D maps of source row and column indices to get a list of coordinates
    r_coords_flat = src_row_indices_map.flatten()  # Shape: [out_h * out_w]
    c_coords_flat = src_col_indices_map.flatten()  # Shape: [out_h * out_w]

    # Stack them to get [N_sampled, 2] where each row is [original_row_index, original_column_index]
    sampled_original_rc_coords_flat = torch.stack((r_coords_flat, c_coords_flat), dim=-1)

    # --- 3. Calculate 1D flat source indices ---
    flat_original_indices = r_coords_flat * orig_w + c_coords_flat  # Shape: [orig_h * orig_w]

    return sampled_original_rc_coords_flat, flat_original_indices, src_row_indices_map, src_col_indices_map


def low_res_depth_to_3d_points(
        low_res_depth_map: torch.Tensor,  # Expected shape: [bs, H_out, W_out]
        original_camera_intrinsics: torch.Tensor,  # Expected shape: [bs, 3, 3]
        original_image_height: int,
        original_image_width: int
) -> torch.Tensor:
    bs, out_h, out_w = low_res_depth_map.shape
    device = low_res_depth_map.device

    # How much the original image was scaled down to get the low-res depth map
    scale_h = float(original_image_height) / float(out_h)
    scale_w = float(original_image_width) / float(out_w)

    # Scale intrinsics to match the low-resolution grid
    fx = (original_camera_intrinsics[:, 0, 0] / scale_w).view(bs, 1, 1)  # [bs, 1, 1]
    fy = (original_camera_intrinsics[:, 1, 1] / scale_h).view(bs, 1, 1)  # [bs, 1, 1]
    cx = (original_camera_intrinsics[:, 0, 2] / scale_w).view(bs, 1, 1)  # [bs, 1, 1]
    cy = (original_camera_intrinsics[:, 1, 2] / scale_h).view(bs, 1, 1)  # [bs, 1, 1]

    # Create coordinate grid for the low-resolution depth map
    v_low_coords_row = torch.arange(out_h, device=device, dtype=torch.float32)  # [out_h]
    u_low_coords_col = torch.arange(out_w, device=device, dtype=torch.float32)  # [out_w]
    v_low_map, u_low_map = torch.meshgrid(v_low_coords_row, u_low_coords_col, indexing='ij')
    u_map = u_low_map[None].expand(bs, -1, -1)  # [B, H, W]
    v_map = v_low_map[None].expand(bs, -1, -1)  # [B, H, W]

    # Perform Back-Projection using Scaled Intrinsics and Low-Res Coordinates
    # Z_world is the depth value from the low-resolution map
    z = low_res_depth_map  # [B, H, W]
    x = (u_map - cx) * z / fx  # [B, H, W]
    y = (v_map - cy) * z / fy  # [B, H, W]

    # Combine into Point Cloud and Reshape
    # points_3d_grid shape: [bs, out_h, out_w, 3] (X, Y, Z order)
    points_3d_grid = torch.stack([x, y, low_res_depth_map], dim=-1)

    # Reshape to [bs, H_out * W_out, 3]
    point_cloud = points_3d_grid.reshape(bs, -1, 3)

    return point_cloud


class GlobalScaleMLP(nn.Module):
    def __init__(self, in_channels=96):
        super(GlobalScaleMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # get one scale factor and one offset factor
        )

    def forward(self, fused_feats):
        # fused feature： [B, num_anchor, 96]
        # mean-pooling
        scale = self.mlp(fused_feats)  # [B, num anchor, 2]
        scale_factor = (scale[..., 0] + 1e-3).mean(dim=1)
        offset_factor = (scale[..., 1] + 1e-3).mean(dim=1)

        return scale_factor, offset_factor,   # [B, 1]


class DPTHead(nn.Module):
    def __init__(self, input_channels: int = 96, output_channels: int = 1):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(input_channels, input_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(input_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels // 2, output_channels, kernel_size=1)
        )

        self.depth_activation = nn.Softplus()

    def forward(self, fused_features: torch.Tensor) -> torch.Tensor:
        depth_map = self.conv_block(fused_features)
        if hasattr(self, 'depth_activation'):
            depth_map = self.depth_activation(depth_map)
        return depth_map  # [b, 1, h, w]


@MODELS.register_module()
class DepthBranchLifter(CustomSerialization):
    def __init__(
            self,
            embed_dims,  # 96
            sample_shape: list,  # [30, 40]
            coding_param: dict,
            scale_mode='ss_sigmoid',
            sample_mode='z-order',  # 'nearest'
            fine_tune_mode='scale',
            noise_intensity=0.015,
            flag_depthbranch=True,
            semantic_dim=0,  # 13
            image_size=(480, 640),  # [h, w]
            supervision=True,
            include_opa=True,
            save_da=None,

            ffn=None,
            proj_layer=None,
            norm_layer=None,
            ms_feat_fuse_layer=None,
            learnable_regulate_factor=False,
            operation_order: Optional[List[str]] = None,
    ):
        super().__init__(**coding_param)
        self.save_da = save_da
        self.embed_dims = embed_dims
        self.image_size = image_size  # [h, w]
        self.supervision = supervision
        self.sample_mode = sample_mode
        self.sample_shape = sample_shape
        self.semantic_dim = semantic_dim
        self.fine_tune_mode = fine_tune_mode
        self.noise_intensity = noise_intensity
        self.flag_depthbranch = flag_depthbranch
        self.include_opa = include_opa
        self.scale_mode = scale_mode
        self.fine_tune_lifter_count = {}

        if operation_order is None:
            operation_order = [
                "proj",
                "fusion",
                "ffn",
                "norm",
            ]
        self.operation_order = operation_order
        if learnable_regulate_factor:
            self.regulate_factor = nn.Parameter(torch.tensor(1.8))  # initialize it as 2.0
        else:
            self.regulate_factor = 1.2

        def build(cfg):
            if cfg is None:
                return None
            return MODELS.build(cfg)

        self.op_config_map = {
            "ffn": ffn,
            "proj": proj_layer,
            "norm": norm_layer,
            "fuse": ms_feat_fuse_layer,
        }
        self.layers = nn.ModuleList(
            [
                build(self.op_config_map.get(op, None))
                for op in self.operation_order
            ]
        )

        # self.norm_layer = nn.LayerNorm(embed_dims)
        # if ms_feat_fuse_layer is not None:
        #     self.ms_feat_fuse_layer = MODELS.build(ms_feat_fuse_layer)
        if self.fine_tune_mode == 'scale':
            self.depth_head = GlobalScaleMLP(embed_dims)
        elif self.fine_tune_mode == 'map':
            # self.depth_decoder_cnn = DepthDecoderCNN(embed_dims, 1)
            self.depth_head = DPTHead(embed_dims, 1)
        else:
            raise ValueError(f"Invalid fine tune mode: {self.fine_tune_mode}, use 'scale' or 'map'")

    def init_weights(self):
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def forward(
            self,
            metas,
            flag_depthanything_as_gt,
            depthnet_output,  # [h, w]
            feature_maps,
            depth_feat,  # [1, 128, h, w]
            depth_conf,  # [1, h, w]
            ELUT: dict = None,
    ):
        """
            30*16 = 480; 40*16 = 640.
        """
        # batch_size = feature_maps[0].shape[0]
        curr_device = feature_maps[0].device
        self._lut_init(curr_device, ELUT)

        curr_name = metas[0]['name']
        self.convert_data = ConvertData(curr_name, self.save_da)

        # parameters
        cam_k = metas[0]['cam_k'].to(torch.float32)
        cam2world = metas[0]['cam2world'].to(torch.float32)  # [bs, 4, 4]
        nyu_vox_range = metas[0]['nyu_vox_range']
        vox_near_world = metas[0]['vox_origin']
        vox_far_world = metas[0]['vox_origin'] + metas[0]['scene_size']  # scene_size: [4.8000, 4.8000, 2.8800]
        vox_range = torch.cat((vox_near_world, vox_far_world), dim=0)
        epsilon = 1e-3
        if self.supervision:
            print(f"\n{64 * '='}\nvox_range: {vox_near_world}~{vox_far_world}, "
                  f"\nvox_cam_range: {nyu_vox_range}")

        if flag_depthanything_as_gt:
            da_pred = metas[0]['depth_gt'].to(torch.float32)
        else:
            da_pred = depthnet_output  # [h, w]
        # h, w = da_pred.shape[:2]  # [480, 640]
        h, w = self.image_size[0], self.image_size[1]  # [480, 640]

        # h_ and w_ should keep the same with that of depthnet_output, so that I can correctly sample depth_feat and depth_conf
        bs_, dim_, h_, w_ = depth_feat.shape
        depth_feat_flatten = depth_feat.permute(0, 2, 3, 1).view(bs_, -1, dim_)  # [1, 1, h*w, 128]
        da_pred_flatten = depthnet_output.view(bs_, -1, 1)  # [B, h * w, 1]
        if depth_conf is not None:
            depth_conf_flatten = depth_conf.view(bs_, -1)  # [1, h*w]
        else:
            depth_conf_flatten = None

        # color
        # color = metas[0]['img_vggt'].squeeze(0).permute(1, 2, 0)  # [h, w, 3]
        # color = metas[0]['img_depthbranch'].squeeze(0).permute(1, 2, 0)  # [h, w, 3]
        # color_permute = color.permute(2, 0, 1).unsqueeze(0)  # [1, 3, h, w]
        # color_reshape = F.interpolate(color_permute, size=(h, w), mode='bilinear', align_corners=False)
        # color_permute = color_reshape.squeeze(0).permute(1, 2, 0)  # [new_h, new_w, 3]
        # color_flatten = color_permute.view(-1, 3)  # [hw, 3]

        if self.supervision:
            color = metas[0]['img_depthbranch']  # [B, 3, h, w]
            color_reshape = F.interpolate(color, size=(h, w), mode='bilinear', align_corners=False)  # [B, 3, new_h, new_w]
            color_permute = color_reshape.permute(0, 2, 3, 1)  # [N, new_h, new_w, 3]
            color_flatten = color_permute.view(bs_, -1, 3)  # [B, hw, 3]
        else:
            color_flatten = None

        if self.sample_mode == 'nearest':
            # sampled_points_2d: [sample h * sample w, 2]; flatten_index_map: [sample h * sample w]; u_indices_map: [sample h, sample w]; v_indices_map: [sample h, sample w]
            sampled_points_2d, flatten_index_map, u_indices_map, v_indices_map = (
                calculate_structured_downsample_indices(h, w, self.sample_shape[0], self.sample_shape[1]))
            depth_feat_sampled = depth_feat_flatten[:, flatten_index_map]  # [1, num anchor, 128] or [1, num anchor, 64]
            depth_pred_sampled = da_pred_flatten[:, flatten_index_map]  # [B, num anchor, 1]
            if depth_conf_flatten is not None:
                depth_conf_sampled = depth_conf_flatten[:, flatten_index_map]  # [1, num anchor]
            else:
                depth_conf_sampled = None

            if self.supervision:
                color_sampled = color_flatten[:, flatten_index_map]  # [B, num, 3]
            else:
                color_sampled = None
        elif self.sample_mode == 'z-order':  # 3D downsample method
            # cam anchors
            da_pts_cam_t = self.pix2cam(da_pred, cam_k)  # [H*W, 3]
            curr_gaussian_cam_mask = self.voxel_cam_filter(da_pts_cam_t, nyu_vox_range)
            curr_cam_anchor = da_pts_cam_t[curr_gaussian_cam_mask]  # [unmasked_num, 3]

            # world anchors
            da_pts_world_t = self.cam2world(curr_cam_anchor, cam2world)  # [hw, 3]
            curr_world_anchor = da_pts_world_t  # [hw, 3]

            # z-order encoding
            curr_encode_order, curr_encode_unique_mask, temp_point = self.pos_encode(curr_world_anchor)  # self.curr_point['code']
            self.curr_point['code'] = temp_point['code']
            depth_feat_sampled = depth_feat_flatten[:, curr_encode_order[0]][:, curr_encode_unique_mask[0]]  # [1, num anchor, 128]
            depth_pred_sampled = da_pred_flatten[:, curr_encode_order[0]][:, curr_encode_unique_mask[0]]  # [1, num anchor, 128]
            if depth_conf_flatten is not None:
                depth_conf_sampled = depth_conf_flatten[:, curr_encode_order[0]][:, curr_encode_unique_mask[0]]  # [1, num anchor]
            else:
                depth_conf_sampled = None

            # curr_encode_order [1, anchor num]; curr_encode_unique_mask [1, unique anchor num]
            da_pts_world_masked = curr_world_anchor[curr_encode_order[0]][curr_encode_unique_mask[0]]  # sort and mask, [num, 3]
            da_pts_cam_masked = curr_cam_anchor[curr_encode_order[0]][curr_encode_unique_mask[0]]  # sort and mask, [num, 3]
            color_sampled = color_flatten[curr_encode_order[0]][curr_encode_unique_mask[0]]
            flatten_index_map = curr_encode_order[0][curr_encode_unique_mask[0]]

            # generate 2d sampling points
            sampled_points_2d = project_points(da_pts_cam_masked, cam_k)  # [num anchor, 2]
        else:
            flatten_index_map = None
            sampled_points_2d = None
            raise ValueError(f"sample mode should be either 'z-order' or 'nearest', but you give {self.sample_mode}. ")

        # cam anchors
        # da_pts_cam_t = self.pix2cam(da_pred, cam_k)  # [hw, 3]
        # world anchors
        # da_pts_world_t = self.cam2world(da_pts_cam_t, cam2world)  # [hw, 3]

        # curr_cam_anchor_down = pcd_voxel_downsample(curr_cam_anchor, voxel_size=0.01)
        # da_pts_cam_2d = project_points(curr_cam_anchor_down.unsqueeze(0), cam_k)
        # da_pts_cam_2d = project_points(da_pts_cam_masked.unsqueeze(0), cam_k)
        # sparse_image = render_sampled_points_to_image(da_pts_cam_masked, color_sampled, cam_k, h, w)

        # self.convert_data.convert_rgb(color, 'rgb')
        # self.convert_data.convert_rgb(sparse_image, '2d')
        # self.convert_data.convert_depth(da_pred, 'depth')
        # self.convert_data.convert_da_pts_color(da_pts_world_t, color_flatten, 'coz_world')

        # adjust initial depth
        inference_feature = depth_feat_sampled  # [1, 128, h, w], or [1, num anchor, 128]
        conf_map = depth_conf_sampled  # [1, h, w] or [1, num anchor] or None

        if self.flag_depthbranch:
            for i, op in enumerate(self.operation_order):
                if op == 'proj' or op == 'norm' or op == 'ffn':
                    inference_feature = self.layers[i](inference_feature)
                if op == 'fuse':
                    start_fuse_toc = time.time()
                    inference_feature = self.layers[i](
                        # da_pts_cam_masked.unsqueeze(0),
                        # sampled_points_2d.unsqueeze(0),  # [1, num anchor, 2]
                        sampled_points_2d.unsqueeze(0).repeat(bs_, 1, 1),  # [B, num anchor, 2]
                        conf_map,
                        inference_feature,
                        feature_maps,
                        metas)  # [1, num anchor, 96]
                    self.fine_tune_lifter_count['fuse_toc'] = time.time() - start_fuse_toc
                    if self.supervision:
                        print(f":: multi-scale fusion cost time: {self.fine_tune_lifter_count['fuse_toc']:.4f}s")

            if self.fine_tune_mode == 'scale':
                scale_factor, offset_factor = self.depth_head(inference_feature)  # [1, 1]

                scale_factor = scale_factor.squeeze(0)  # [1]
                if self.scale_mode == 'ss_sigmoid':
                    scale_factor = sign_shaft_sigmoid(scale_factor)
                elif self.scale_mode == 'tanh':
                    scale_factor = torch.tanh(scale_factor)
                else:
                    raise ValueError(f"invalid scale mode {self.scale_mode}, use 'ss_sigmoid' or 'tanh'. ")
                scale_factor = scale_factor * self.regulate_factor  # FIXME, baseline: 1.2, 1.8

                # update
                # print(f"scale factor: {scale_factor}")
                da_pred_scaled = da_pred.unsqueeze(0) * (1 + scale_factor)  # [1, h, w]

                # cam anchors & world anchors
                da_pts_cam_scaled = self.pix2cam(da_pred_scaled, cam_k)  # [H*W, 3]
                da_pts_world_scaled = self.cam2world(da_pts_cam_scaled, cam2world)  # [hw, 3]

                da_pts_cam_sampled_scaled = da_pts_cam_scaled[flatten_index_map]  # [num, 3]
                da_pts_world_sampled_scaled = self.cam2world(da_pts_cam_sampled_scaled, cam2world)  # [num, 3]

            elif self.fine_tune_mode == 'map':
                output_feature = inference_feature.permute(0, 2, 1).view(bs_, self.embed_dims, self.sample_shape[0], self.sample_shape[1])  # [B, 96, 30, 40]
                refined_depth_map = self.depth_head(output_feature)  # [B, 1, 480, 640]
                da_pred_scaled = refined_depth_map.permute(0, 2, 3, 1).reshape(bs_, self.sample_shape[0], self.sample_shape[1])  # bs, [h, w]

                da_pts_cam_sampled_scaled = low_res_depth_to_3d_points(da_pred_scaled, cam_k, h_, w_)  # [bs, num, 3]
                da_pts_world_sampled_scaled = self.cam2world(da_pts_cam_sampled_scaled, cam2world)  # [bs, num, 3]
            else:
                da_pred_scaled = None
                da_pts_cam_sampled_scaled = None
                da_pts_world_sampled_scaled = None
                raise ValueError(f"Invalid fine tune mode {self.fine_tune_mode}, use 'map or 'scale''")
        else:
            # [bs, num anchor, 1] --> [bs, h, w]
            da_pred_scaled = depth_pred_sampled.reshape(bs_, self.sample_shape[0], self.sample_shape[1])
            da_pts_cam_sampled_scaled = low_res_depth_to_3d_points(da_pred_scaled, cam_k, h_, w_)  # [bs, num, 3]
            da_pts_world_sampled_scaled = self.cam2world(da_pts_cam_sampled_scaled, cam2world)
            da_pred_scaled = da_pred_scaled.reshape(bs_, self.sample_shape[0], self.sample_shape[1])  # [bs, h, w]

        H, W = self.sample_shape
        mask_hw = torch.zeros((H, W), dtype=torch.bool, device=da_pred_scaled.device)
        ignore_idx = int(H / 30)
        h_v = H - ignore_idx
        w_v = W - ignore_idx
        mask_hw[ignore_idx:, ignore_idx:] = True  # [h, w]
        mask_flat = mask_hw.flatten()  # [hw]

        da_pts_cam_valid = da_pts_cam_sampled_scaled[:, mask_flat]  # [b, num, 3]
        da_pts_world_valid = da_pts_world_sampled_scaled[:, mask_flat]  # [b, num, 3]

        if self.supervision:
            color_sampled_valid = color_sampled[:, mask_flat]  # [b, num, 3]
            color_sampled_valid = self.convert_data.restore_map(color_sampled_valid) / 255.0
            self.convert_data.convert_da_pts_color(da_pts_world_valid, color_sampled_valid, "csz_world")
            self.convert_data.convert_da_pts_color(da_pts_cam_valid, color_sampled_valid, "csz_cam")

        da_pred_scaled_valid = da_pred_scaled[:, mask_hw].reshape(bs_, h_v, w_v)  # [bs, h', w']
        flatten_index_map_valid = flatten_index_map[mask_flat.cpu()]  # [num anchor]
        inference_feature_valid = inference_feature[:, mask_flat]  # [bs, num anchor, 96]
        if conf_map is not None:
            conf_map_valid = conf_map[:, mask_hw]
        else:
            conf_map_valid = None

        # self.convert_data.convert_depth(da_pred_scaled.squeeze(0), 'depth_pred')
        # self.convert_data.convert_da_pts_nocolor(da_pts_world_scaled, "world")
        # self.convert_data.convert_da_pts_nocolor(da_pts_world_sampled_scaled.squeeze(0), "z_world")

        # return da_pred_scaled, da_pts_cam_sampled_scaled.squeeze(0), da_pts_world_sampled_scaled.squeeze(0), flatten_index_map, inference_feature, conf_map
        return da_pred_scaled_valid, da_pts_cam_valid, da_pts_world_valid, flatten_index_map_valid, inference_feature_valid, conf_map_valid


"""

def downsample_structured_with_indices(
        original_data_grid: torch.Tensor,
        out_h: int,
        out_w: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    # --- Input Validation ---
    if not (original_data_grid.ndim == 3 or original_data_grid.ndim == 2):
        raise ValueError(
            "original_data_grid must be a 2D or 3D tensor (H, W, C) or (H, W)."
            f" Got shape: {original_data_grid.shape}"
        )

    orig_h = original_data_grid.shape[0]
    orig_w = original_data_grid.shape[1]
    device = original_data_grid.device

    if out_h <= 0 or out_w <= 0:
        raise ValueError(f"Output dimensions out_h ({out_h}) and out_w ({out_w}) must be positive.")

    # Note: This logic primarily targets downsampling or same-size operations.
    # If out_h > orig_h or out_w > orig_w (upsampling), 'nearest' logic still picks one source pixel.
    # F.interpolate provides more sophisticated upsampling modes if needed.
    if out_h > orig_h or out_w > orig_w:
        pass  # Allow 'nearest' upsampling behavior if user intends it.

    # --- 1. Calculate source indices (mimicking F.interpolate(mode='nearest')) ---

    # Create coordinates for the output grid
    # y_out_coords will have shape [out_h, 1]
    # x_out_coords will have shape [1, out_w]
    y_out_coords = torch.arange(out_h, device=device, dtype=torch.float32).view(out_h, 1)
    x_out_coords = torch.arange(out_w, device=device, dtype=torch.float32).view(1, out_w)

    # Calculate scaling factors (input_dim / output_dim)
    scale_h = float(orig_h) / float(out_h)
    scale_w = float(orig_w) / float(out_w)

    # Compute source row indices for each output row, then expand to full map
    # Equivalent to: src_y = floor(y_out * scale_h)
    src_row_indices = torch.floor(y_out_coords * scale_h).long()
    src_row_indices_map = src_row_indices.expand(-1, out_w)  # Shape: [out_h, out_w]

    # Compute source column indices for each output column, then expand to full map
    # Equivalent to: src_x = floor(x_out * scale_w)
    src_col_indices = torch.floor(x_out_coords * scale_w).long()
    src_col_indices_map = src_col_indices.expand(out_h, -1)  # Shape: [out_h, out_w]

    # Clamp indices to ensure they are within the bounds of the original grid
    src_row_indices_map = torch.clamp(src_row_indices_map, 0, orig_h - 1)
    src_col_indices_map = torch.clamp(src_col_indices_map, 0, orig_w - 1)

    # --- 2. Sample data using the calculated 2D indices ---
    if original_data_grid.ndim == 3:  # Input is (H, W, C)
        data_sampled = original_data_grid[src_row_indices_map, src_col_indices_map, :]
    else:  # Input is (H, W)
        data_sampled = original_data_grid[src_row_indices_map, src_col_indices_map]

    # --- 3. Calculate flat source indices ---
    # These indices can be used if original_data_grid is flattened to [orig_h * orig_w, ...]
    # The formula for row-major flat index is: row_index * num_cols_in_original + col_index
    flat_src_indices_map = src_row_indices_map * orig_w + src_col_indices_map  # Shape: [out_h, out_w]
    flat_src_indices = flat_src_indices_map.flatten()  # Shape: [out_h * out_w]

    return data_sampled, src_row_indices_map, src_col_indices_map, flat_src_indices


class DepthDecoderCNN(nn.Module):
    def __init__(self, input_channels: int = 96, output_channels: int = 1):
        super().__init__()

        # Upsampling stages: 30x40 -> 60x80 -> 120x160 -> 240x320 -> 480x640
        # Each stage typically upsamples by 2x and then refines with convolutions.

        # Stage 1: 30x40 -> 60x80
        # Channels: input_channels (96) -> 64
        self.upconv1 = nn.ConvTranspose2d(input_channels, 64, kernel_size=4, stride=2,
                                          padding=1)  # Output H,W = 2*H_in, 2*W_in
        self.conv1_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.relu1_2 = nn.ReLU(inplace=True)

        # Stage 2: 60x80 -> 120x160
        # Channels: 64 -> 32
        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.conv2_1 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(32)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn2_2 = nn.BatchNorm2d(32)
        self.relu2_2 = nn.ReLU(inplace=True)

        # Stage 3: 120x160 -> 240x320
        # Channels: 32 -> 16
        self.upconv3 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.conv3_1 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(16)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False)
        self.bn3_2 = nn.BatchNorm2d(16)
        self.relu3_2 = nn.ReLU(inplace=True)

        # Stage 4: 240x320 -> 480x640
        # Channels: 16 -> output_channels (1 for depth)
        self.upconv4 = nn.ConvTranspose2d(16, output_channels, kernel_size=4, stride=2, padding=1)
        # self.upconv4 = nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1)
        # self.conv4_1 = nn.Conv2d(8, 8, kernel_size=3, padding=1, bias=False)
        # self.bn4_1 = nn.BatchNorm2d(8)
        # self.relu4_1 = nn.ReLU(inplace=True)
        # self.final_conv = nn.Conv2d(8, output_channels, kernel_size=3, padding=1) # Final prediction layer

        # Optional: A final activation for the depth map, e.g., ReLU if depth is non-negative
        # Or no activation if the loss function handles the range.
        # For depth, often no activation or a ReLU is used.
        # self.final_activation = nn.ReLU(inplace=True) # if depth must be >= 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Stage 1
        x = self.upconv1(x)  # [B, 64, 60, 80]
        x = self.relu1_1(self.bn1_1(self.conv1_1(x)))
        x = self.relu1_2(self.bn1_2(self.conv1_2(x)))

        # Stage 2
        x = self.upconv2(x)  # [B, 32, 120, 160]
        x = self.relu2_1(self.bn2_1(self.conv2_1(x)))
        x = self.relu2_2(self.bn2_2(self.conv2_2(x)))

        # Stage 3
        x = self.upconv3(x)  # [B, 16, 240, 320]
        x = self.relu3_1(self.bn3_1(self.conv3_1(x)))
        x = self.relu3_2(self.bn3_2(self.conv3_2(x)))

        # Stage 4 & Final Output
        x = self.upconv4(x)  # [B, 1, 480, 640]
        # x = self.relu4_1(self.bn4_1(self.conv4_1(x)))
        # x = self.final_conv(x)

        # if hasattr(self, 'final_activation'):
        #     x = self.final_activation(x)

        return x

"""