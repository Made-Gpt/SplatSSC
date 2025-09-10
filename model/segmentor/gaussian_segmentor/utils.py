import cv2
import copy
import time
import torch
import numpy as np
import pandas as pd
import open3d as o3d
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch.nn.functional as F
from matplotlib.collections import PatchCollection

from dataclasses import dataclass
from plyfile import PlyData, PlyElement
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from math import sqrt, pi


color_list = [
    [0.0, 0.0, 0.0],      # 0: Background
    [0.84, 0.48, 0.48],   # 1: Ceiling
    [0.48, 0.84, 0.48],   # 2: Floor
    [0.48, 0.48, 0.84],   # 3: Wall
    [0.84, 0.84, 0.48],   # 4: Window
    [0.84, 0.48, 0.84],   # 5: Chair
    # [0.48, 0.84, 0.84],   # 6: Bed
    [0.63, 0.87, 0.96],   # 6: Bed
    [0.79, 0.68, 0.85],   # 7: Sofa
    [0.96, 0.72, 0.48],   # 8: Table
    [0.6, 0.72, 0.48],    # 9: TVs
    [0.48, 0.72, 0.72],   # 10: Furniture
    # [0.48, 0.60, 0.96],  # 11: Object
    [0.32, 0.54, 0.78],   # 11: Object
    [0.96, 0.96, 0.96],   # 12: Empty
]


lines = [
    [0, 1], [1, 3], [3, 2], [2, 0],
    [4, 5], [5, 7], [7, 6], [6, 4],
    [0, 4], [1, 5], [2, 6], [3, 7]
]


def rigid_transform(xyz, transform):
    """Applies a rigid transform to an (N, 3) pointcloud."""
    xyz_h = np.hstack([xyz, np.ones((xyz.shape[0], 1), dtype=np.float32)])
    xyz_t_h = np.dot(transform, xyz_h.T).T
    return xyz_t_h[:, :3]


def occ2world(occs, vox_origin, vox_size):
    """Convert voxel grid coordinates to world coordinates."""
    bs, H, W, D = occs.shape  # [1, 60, 60, 36]
    x_coords = np.linspace(vox_origin[0] + vox_size / 2,
                           vox_origin[0] + (W - 0.5) * vox_size, W)
    y_coords = np.linspace(vox_origin[1] + vox_size / 2,
                           vox_origin[1] + (H - 0.5) * vox_size, H)
    z_coords = np.linspace(vox_origin[2] + vox_size / 2,
                           vox_origin[2] + (D - 0.5) * vox_size, D)
    xx, yy, zz = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    pts = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)

    labels = occs.reshape(-1).astype(np.int32)
    mask = (labels != 12) & (labels != 0)
    colors = np.array([color_list[label] for label in labels])
    pts = pts[mask]  # [num, 3]
    colors = colors[mask]  # [num, 3]

    return pts, colors


def transform_vox_range(vox_range, world2cam):
    # vox_range: np, world2cam: np
    x_min, y_min, z_min, x_max, y_max, z_max = vox_range
    world_corners = np.array([
        [x_min, y_min, z_min, 1],  # 000
        [x_min, y_min, z_max, 1],  # 001
        [x_min, y_max, z_min, 1],  # 010
        [x_min, y_max, z_max, 1],  # 011
        [x_max, y_min, z_min, 1],  # 100
        [x_max, y_min, z_max, 1],  # 101
        [x_max, y_max, z_min, 1],  # 110
        [x_max, y_max, z_max, 1],  # 111
    ])

    cam_corners = (world2cam @ world_corners.T).T  # (8, 4)
    cam_corners = cam_corners[:, :3]

    cam_x_min, cam_y_min, cam_z_min = cam_corners.min(axis=0)
    cam_x_max, cam_y_max, cam_z_max = cam_corners.max(axis=0)

    cam_vox_range = np.array([cam_x_min, cam_y_min, cam_z_min, cam_x_max, cam_y_max, cam_z_max])
    return cam_vox_range


def point2voxel(grid_coord, points, origin, boundary, grid_size):
    # boundary and origin shape should both be [3]
    grid_coord = grid_coord.squeeze()  # [num, 3]
    points_int = torch.round((points.detach() - origin) / grid_size).to(torch.int)  # [num, 3]
    mask = (points_int >= 0) & (points_int <= boundary.unsqueeze(0))
    points_int_masked = points_int[mask]  # [num, 3]

    return points_int_masked


def project_points(key_pts, cam_k):
    # key_pts shape should be [1, num, 3] or [num, 3]
    f_l_x, f_l_y = cam_k[0, 0], cam_k[1, 1]
    c_x, c_y = cam_k[0, 2], cam_k[1, 2]
    points_2d_x = f_l_x * key_pts[..., 0] / key_pts[..., 2] + c_x
    points_2d_y = f_l_y * key_pts[..., 1] / key_pts[..., 2] + c_y
    if len(key_pts.shape) == 3:
        points_2d = torch.stack((points_2d_x, points_2d_y), dim=2)  # [1, num, 2]
    else:
        points_2d = torch.stack((points_2d_x, points_2d_y), dim=1)  # [num, 2]
    return points_2d


def render_sampled_points_to_image(
        sampled_points_3d: torch.Tensor,
        sampled_colors: torch.Tensor,
        cam_k: torch.Tensor,
        image_height: int,
        image_width: int
) -> torch.Tensor:

    # --- 1. Points and Colors to Project ---
    points_to_project = sampled_points_3d
    colors_to_project = sampled_colors

    # --- 2. Project 3D points to 2D ---
    uv_proj = project_points(points_to_project, cam_k)
    u_proj = uv_proj[..., 0]
    v_proj = uv_proj[..., 1]

    # --- 3. Round 2D coordinates to integer pixel locations ---
    u_pixels = torch.round(u_proj).long()
    v_pixels = torch.round(v_proj).long()

    # --- 4. Filter points that project outside the image boundaries ---
    in_bounds_mask = (u_pixels >= 0) & (u_pixels < image_width) & \
                     (v_pixels >= 0) & (v_pixels < image_height)

    # If all points project out of bounds, return a black image.
    # if not torch.any(in_bounds_mask):
    #     return torch.zeros((image_height, image_width, 3),
    #                        dtype=sampled_colors.dtype,
    #                        device=sampled_points_3d.device)

    final_u_coords = u_pixels[in_bounds_mask]
    final_v_coords = v_pixels[in_bounds_mask]
    final_colors = colors_to_project[in_bounds_mask]

    # --- 5. Create the output image and "paint" the points ---
    output_image = torch.zeros((image_height, image_width, 3),
                               dtype=sampled_colors.dtype,
                               device=sampled_points_3d.device)

    output_image[final_v_coords, final_u_coords, :] = final_colors

    return output_image


gaussian_attributes = [
    'x', 'y', 'z', 'opacity',
    'f_dc_0', 'f_dc_1', 'f_dc_2',
    'scale_0', 'scale_1', 'scale_2',
    'rot_0', 'rot_1', 'rot_2', 'rot_3',
]


class ConvertData:
    def __init__(
            self,
            curr_name,
            save_da,
            cmap=None,
        ):
        if cmap is None:
            self.cmap = cm.get_cmap('Spectral_r')
        else:
            self.cmap = cmap
        self.save_da = save_da

        self.curr_name = curr_name[0]

    def restore_map(self, map):
        norm_map = (map - map.min()) / (map.max() - map.min()) * 255.0
        # norm_map = 255.0 - norm_map
        if isinstance(map, np.ndarray):
            norm_map = norm_map.astype(np.uint8)
        else:
            norm_map = norm_map.to(torch.float32)
        return norm_map

    def apply_cmap(self, map):
        ''' input numpy.ndary, only have 2 dimensions'''
        norm_map = self.restore_map(map)
        color_map = (self.cmap(norm_map)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        return color_map

    # save color image
    def convert_rgb(self, color, name='rgb'):
        # color shape must be [h, w, 3]
        if isinstance(color, torch.Tensor):
            color = color.detach().cpu().numpy()
        h, w, c = color.shape
        color = self.restore_map(color.reshape(-1, 3))
        color_map = color.reshape(h, w, c)
        color_map = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)
        scene_name, file_name = self.curr_name.split('/')[0], self.curr_name.split('/')[1]
        save_rgb = Path(self.save_da+f'_{name}') / scene_name
        save_rgb.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(str(save_rgb / f'rgb_{file_name}.png'), color_map)
        print(f"save rgb_{file_name}.png to {save_rgb}, shape: {color.shape}")
        return color_map

    def convert_original_rgb(self, color, name='ori_rgb'):
        rgb_shape = color.shape
        if len(rgb_shape) == 4:
            if rgb_shape[-1] != 3:  # (b, n, w, h)
                color_map = color.squeeze(0).permute(1, 2, 0).cpu().numpy()
            else:  # (b, w, h, n)
                color_map = color.squeeze(0).cpu().numpy()
        else:
            if rgb_shape[-1] != 3:  # (n, w, h)
                color_map = color.permute(1, 2, 0).cpu().numpy()
            else:  # (w, h, n)
               color_map = color.cpu().numpy()
        scene_name, file_name = self.curr_name.split('/')[0], self.curr_name.split('/')[1]
        save_rgb = Path(self.save_da+f'_{name}') / scene_name
        save_rgb.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(str(save_rgb / f'rgb_{file_name}.png'), color_map)
        print(f"save rgb_{file_name}.png to {save_rgb}, shape: {color_map.shape}")
        return color_map

    # save depth map
    def convert_depth(self, depth_map, name='depth'):
        if isinstance(depth_map, torch.Tensor):
            depth = depth_map.detach().cpu().numpy()
        else:
            depth = depth_map
        if len(depth.shape) == 3:  # [bs, num, 1]
            depth = depth.squeeze(0)

        depth = self.apply_cmap(depth)

        scene_name, file_name = self.curr_name.split('/')[0], self.curr_name.split('/')[1]
        save_depth = Path(self.save_da+f'_{name}') / scene_name
        save_depth.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(str(save_depth / f'depth_{file_name}.png'), depth)
        print(f"save depth_{file_name}.png to {save_depth}, shape: {depth.shape}")

    # save depth map as numpy array
    def convert_depth_np(self, depth_map, name='depth_np'):
        if isinstance(depth_map, torch.Tensor):
            depth = depth_map.detach().cpu().numpy()
        else:
            depth = depth_map
        depth = self.apply_cmap(depth)

        scene_name, file_name = self.curr_name.split('/')[0], self.curr_name.split('/')[1]
        save_depth = Path(self.save_da+f'_{name}') / scene_name
        save_depth.mkdir(exist_ok=True, parents=True)
        np.save(str(save_depth / f'depth_{file_name}.npy'), depth)
        print(f"save depth_{file_name}.npy to {save_depth}, shape: {depth.shape}")

    # save confidence map
    def convert_conf(self, conf_map, name='conf'):
        conf = conf_map.detach().cpu().numpy()  # [1, 1, h, w]
        conf_map = self.apply_cmap(conf)

        scene_name, file_name = self.curr_name.split('/')[0], self.curr_name.split('/')[1]
        save_conf = Path(self.save_da+f'_{name}') / scene_name
        save_conf.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(str(save_conf / f'conf_{file_name}.png'), conf_map)
        np.save(str(save_conf / f'conf_{file_name}.npy'), conf)
        print(f"save conf_{file_name}.png/npy to {save_conf}")

    def convert_camera_info(self, pose, intrinsic, name='camera'):
        if isinstance(pose, torch.Tensor):
            pose = pose.detach().cpu().numpy()
        if isinstance(intrinsic, torch.Tensor):
            intrinsic = intrinsic.detach().cpu().numpy()

        scene_name, file_name = self.curr_name.split('/')[0], self.curr_name.split('/')[1]
        save_cam = Path(self.save_da + f'_{name}') / scene_name
        save_cam.mkdir(exist_ok=True, parents=True)
        np.save(str(save_cam / f'pose_{file_name}.npy'), pose)
        print(f"Saved pose_{file_name}.npy to {save_cam}, shape: {pose.shape}")
        np.save(str(save_cam / f'intrinsic_{file_name}.npy'), intrinsic)
        print(f"Saved intrinsic_{file_name}.npy to {save_cam}, shape: {intrinsic.shape}")

    # useless
    def convert_vox_range(self, vox_range, name='vox_range'):
        # build bbox
        vox_range_cpu = vox_range.detach().cpu().numpy()
        box_min = vox_range_cpu[:3]  # [x_min, y_min, z_min]
        box_max = vox_range_cpu[3:]  # [x_max, y_max, z_max]
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=box_min, max_bound=box_max)
        bbox.color = (1, 0, 0)
        bbox_points = o3d.utility.Vector3dVector(bbox.get_box_points())
        line_set = o3d.geometry.LineSet()
        line_set.points = bbox_points
        line_set.lines = o3d.utility.Vector2iVector(lines)
        # save
        scene_name, file_name = self.curr_name.split('/')[0], self.curr_name.split('/')[1]
        save_pcd = Path(self.save_da+f'_{name}') / scene_name
        save_pcd.mkdir(exist_ok=True, parents=True)
        save_path = str(save_pcd / f'bbox_{file_name}.ply')
        o3d.io.write_line_set(save_path, line_set)
        print(f"Saved bounding box as bbox_{file_name}.ply to {save_pcd}, vox_range: {vox_range}")

    # save point cloud without color
    def convert_da_pts_nocolor(self, pts, name='cam'):
        if isinstance(pts, torch.Tensor):
            pts_cam = pts.detach().cpu().numpy()  # (num, 3)
        else:
            pts_cam = copy.deepcopy(pts)  # (num, 3)
        # Generate PCD
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_cam)
        scene_name, file_name = self.curr_name.split('/')[0], self.curr_name.split('/')[1]
        save_pcd = Path(self.save_da+f'_{name}') / scene_name
        save_pcd.mkdir(exist_ok=True, parents=True)
        save_path = str(save_pcd / f'pcd_{file_name}.ply')
        o3d.io.write_point_cloud(save_path, pcd)
        print(f"save pcd_{file_name}.ply to {save_pcd}, size: {pts_cam.shape}")

    # save point cloud with color
    def convert_da_pts_color(self, pts, color, name='cam', sub_name=None):

        print(f"pts: {pts.shape}, {pts.min():.4f}~{pts.max():.4f}\n"
              f"color: {color.shape}, {color.min():.4f}~{color.max():.4f}")

        if isinstance(pts, torch.Tensor):
            pts_np = pts.detach().cpu().numpy()  # (num, 3)
        else:
            pts_np = copy.deepcopy(pts)  # (num, 3)

        if isinstance(color, torch.Tensor):
            color_np = color.detach().cpu().numpy()  # (num, 3)
        else:
            color_np = copy.deepcopy(color)  # (num, 3)

        if len(pts_np.shape) == 3:
            pts_np = pts_np.squeeze(0)  # (num, 3)

        if len(color_np.shape) == 3:
            color_np = color_np.squeeze(0)  # (num, 3)

        # Generate PCD
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_np)
        pcd.colors = o3d.utility.Vector3dVector(color_np)
        scene_name, file_name = self.curr_name.split('/')[0], self.curr_name.split('/')[1]
        save_pcd = Path(self.save_da+f'_{name}') / scene_name
        save_pcd.mkdir(exist_ok=True, parents=True)
        if sub_name is not None:
            middle_name = f'{file_name}_{sub_name}'
        else:
            middle_name = file_name
        save_path = str(save_pcd / f'pcd_{middle_name}.ply')
        o3d.io.write_point_cloud(save_path, pcd)
        print(f"save pcd_{middle_name}.ply to {save_pcd}, size: {pts_np.shape}")

    # save 2d point cloud without color
    def convert_da_pts_2d_nocolor(self, pts_2d, name='2d'):
        if isinstance(pts_2d, torch.Tensor):
            pts_cam = pts_2d.detach().cpu().numpy()  # (num, 2)
        else:
            pts_cam = copy.deepcopy(pts_2d)  # (num, 2)
        pts_cam = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)  # (num, 3)
        # Generate PCD
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_cam)
        scene_name, file_name = self.curr_name.split('/')[0], self.curr_name.split('/')[1]
        save_pcd = Path(self.save_da+f'_{name}') / scene_name
        save_pcd.mkdir(exist_ok=True, parents=True)
        save_path = str(save_pcd / f'pcd_{file_name}.ply')
        o3d.io.write_point_cloud(save_path, pcd)
        print(f"save pcd_{file_name}.ply to {save_pcd}, size: {pts_cam.shape}")

    # sampling visualization
    def convert_occ_image_sampling(self, pts_2d, image, name='ref'):
        # --- 1. Data Preparation ---
        if isinstance(pts_2d, torch.Tensor):
            pts_np = pts_2d.detach().cpu().numpy()
        else:
            pts_np = copy.deepcopy(pts_2d)

        if pts_np.shape[1] == 3:
            pts_np = pts_np[:, :2]

        if isinstance(image, torch.Tensor):
            image_np = image.permute(1, 2, 0).detach().cpu().numpy() if image.shape[
                                                                            0] == 3 else image.detach().cpu().numpy()
        else:
            image_np = copy.deepcopy(image)

        if image_np.dtype in [np.float32, np.float64]:
            image_np = (np.clip(image_np, 0, 1) * 255).astype(np.uint8) if image_np.max() <= 1.0 else np.clip(image_np, 0, 255).astype(
                np.uint8)

        # --- 2. Reshape and Filter Data ---
        num_total_points = pts_np.shape[0]
        if num_total_points == 0 or num_total_points % 7 != 0:
            print(f"Warning: Total points ({num_total_points}) is zero or not divisible by 7. Skipping plot.")
            return

        num_clusters = num_total_points // 7
        pts_clustered = pts_np.reshape(num_clusters, 7, 2)

        valid_mask_per_cluster = ((pts_clustered >= 0.0) & (pts_clustered <= 1.0)).all(axis=(1, 2))

        if not np.any(valid_mask_per_cluster):
            print("Warning: All points are outside the valid [0,1] coordinate range. No points to plot.")
            return

        pts_clustered = pts_clustered[valid_mask_per_cluster]
        num_valid_clusters = pts_clustered.shape[0]
        print(f"Processing {num_valid_clusters} valid clusters out of {num_clusters} initial clusters.")

        h, w = image_np.shape[:2]
        pts_clustered_pixels = pts_clustered * np.array([w, h])

        # --- 3. Prepare Colors and Scatter Properties (Vectorized) ---
        # Ellipse colors from 'viridis' colormap
        cluster_colors_rgb = cm.get_cmap('viridis', num_valid_clusters)(np.linspace(0, 1, num_valid_clusters))

        # PROBLEM 1 FIX: Lower alpha for ellipses to make them more transparent
        ellipse_rgba_colors = np.hstack([cluster_colors_rgb[:, :3], np.full((num_valid_clusters, 1), 0.23)])

        # Prepare properties for the scatter points
        scatter_points_flat = pts_clustered_pixels.reshape(-1, 2)
        # Make offset points a neutral dark color, and the center point a bright white
        scatter_point_colors = np.repeat(cluster_colors_rgb, 7, axis=0)
        center_indices = np.arange(0, num_valid_clusters * 7, 7)
        scatter_point_colors[center_indices] = [1.0, 1.0, 1.0, 1.0]  # Bright white for centers

        # Add a black edge ONLY to the center points for visibility
        scatter_edge_colors = np.array(['none'] * (num_valid_clusters * 7), dtype=object)
        scatter_edge_colors[center_indices] = 'black'
        scatter_linewidths = np.zeros(num_valid_clusters * 7)
        scatter_linewidths[center_indices] = 0.5

        # --- 4. Create Ellipse Patches (Efficient Loop) ---
        ellipse_patches = []
        for i in range(num_valid_clusters):
            cluster_points = pts_clustered_pixels[i]
            center_point = cluster_points[0]
            offset_points = cluster_points[1:]

            cov = np.cov(offset_points, rowvar=False)

            try:
                eigvals, eigvecs = np.linalg.eigh(cov)
            except np.linalg.LinAlgError:
                continue

            major_eigenvector = eigvecs[:, 1]
            minor_eigenvector = eigvecs[:, 0]

            # Calculate the angle of the major axis.
            angle_rad = np.arctan2(major_eigenvector[1], major_eigenvector[0])
            angle_deg = np.degrees(angle_rad)

            # The width and height are scaled by the standard deviations (sqrt of eigenvalues).
            # A scale factor of 2.0 corresponds to the 2-sigma boundary.
            width = 3.2 * np.sqrt(eigvals[1])
            height = 3.2 * np.sqrt(eigvals[0])

            ellipse = patches.Ellipse(xy=center_point, width=width, height=height, angle=angle_deg)
            ellipse_patches.append(ellipse)

        # --- 5. Plotting with Layers ---
        plt.figure(figsize=(w / 30, h / 30), dpi=150)  # Increased DPI for better quality
        ax = plt.gca()

        # Layer 1: Background Image (zorder=0 by default)
        plt.imshow(image_np)

        # Layer 2: Ellipses
        ellipse_collection = PatchCollection(
            ellipse_patches,
            facecolors=ellipse_rgba_colors,
            edgecolors='black',
            linewidths=0.5,  # Very thin outline for definition
            zorder=2  # Draw ellipses above the background
        )
        ax.add_collection(ellipse_collection)

        # PROBLEM 2 FIX: Layer 3: Scatter points on top
        ax.scatter(
            scatter_points_flat[:, 0],
            scatter_points_flat[:, 1],
            c=scatter_point_colors,
            s=int(w/10),  # Small but visible points
            edgecolors=scatter_edge_colors,
            linewidths=scatter_linewidths,
            zorder=3  # Draw points on top of ellipses
        )

        ax.axis('off')

        # --- 6. Save the Figure ---
        try:
            scene_name, file_name = self.curr_name.split('/')
        except (AttributeError, ValueError):
            scene_name, file_name = 'default_scene', 'default_file'

        save_dir = Path(getattr(self, 'save_da', './results') + f'_{name}') / scene_name
        save_dir.mkdir(exist_ok=True, parents=True)
        save_path = str(save_dir / f'sampling_{file_name}.png')

        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        print(f"Saved frame {file_name} visualization to {save_path}")

    def convert_occ_voxel_pts(self, pts, grid_size, origin_use, name='bin'):
        # pts: [num, 3], tensor, origin_use: [3]
        origin_use = origin_use.to(torch.float32).to(pts.device)
        pts_int = ((pts - origin_use) / grid_size).to(torch.int)
        pts_hit = torch.unique(pts_int, dim=0) * grid_size + origin_use + grid_size / 2  # [num, 3]
        pts_hit = pts_hit.detach().cpu().numpy()
        # generate pcd
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_hit)
        scene_name, file_name = self.curr_name.split('/')[0], self.curr_name.split('/')[1]
        save_pcd = Path(self.save_da + f'_{name}') / scene_name
        save_pcd.mkdir(exist_ok=True, parents=True)
        save_path = str(save_pcd / f'pcd_{file_name}.ply')
        o3d.io.write_point_cloud(save_path, pcd)
        print(f"save pcd_{file_name}.ply to {save_pcd}, size: {pts_hit.shape}")

    def convert_occ_label_color(self, label, voxel_size, voxel_origin, name='label', alpha=1, ignore=[0, 12], matrix=None):
        if len(label.shape) == 5:
            label = torch.argmax(label, dim=1)
        B, H, W, D = label.shape

        if isinstance(label, torch.Tensor):
            label_cpu = label.cpu().numpy().reshape(-1).astype(np.int32)
        else:
            label_cpu = copy.deepcopy(label).reshape(-1).astype(np.int32)

        if isinstance(voxel_origin, torch.Tensor):
            voxel_origin_cpu = voxel_origin.cpu().numpy()
            if len(voxel_origin_cpu.shape) == 2:
                voxel_origin_cpu = voxel_origin_cpu[0]
        elif isinstance(voxel_origin, list):
            voxel_origin_cpu = voxel_origin[0]
        else:
            voxel_origin_cpu = copy.deepcopy(voxel_origin)

        x_coords = np.linspace(voxel_origin_cpu[0] + voxel_size / 2,
                               voxel_origin_cpu[0] + (W - 0.5) * voxel_size, W)
        y_coords = np.linspace(voxel_origin_cpu[1] + voxel_size / 2,
                               voxel_origin_cpu[1] + (H - 0.5) * voxel_size, H)
        z_coords = np.linspace(voxel_origin_cpu[2] + voxel_size / 2,
                               voxel_origin_cpu[2] + (D - 0.5) * voxel_size, D)
        xx, yy, zz = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        pts = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)

        if isinstance(ignore, list):
            mask = np.ones_like(label_cpu, dtype=bool)
            for c in ignore:
                mask &= (label_cpu != c)
        else:
            mask = (label_cpu != ignore)
        colors = np.array([color_list[label] for label in label_cpu])
        # mask empty and unknown voxels
        pts = pts[mask]
        colors = colors[mask] * alpha

        # convert coordinate
        if matrix is not None:
            if isinstance(matrix, torch.Tensor):
                matrix = matrix.detach().cpu().numpy()
            # matrix = matrix.squeeze(0)
            ones = np.ones((1, pts.shape[0]))
            pts_ori_hom = np.vstack([pts.T, ones])  # [4, (H*W)']
            pts_new_hom = matrix @ pts_ori_hom  # [4, (H*W)']
            pts_new = pts_new_hom[:3, :]  # [3, (H*W)']
            pts = pts_new.T  # [(H*W)', 3]

        # save
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        scene_name, file_name = self.curr_name.split('/')[0], self.curr_name.split('/')[1]
        save_pcd = Path(self.save_da+f'_{name}') / scene_name
        save_pcd.mkdir(exist_ok=True, parents=True)
        save_path = str(save_pcd / f'pcd_{file_name}.ply')
        o3d.io.write_point_cloud(save_path, pcd)
        print(f"save pcd_{file_name}.ply to {save_pcd}, size: {pts.shape}")

    def convert_cov_matrix(self, cov, inv_cov, scales, eigenvectors, xyz, name='cov'):
        # xyz shape is [1, num, 3]

        _, num_anchor = cov.shape[:2]
        indices = torch.arange(int(num_anchor/3), int(num_anchor/3)*2)

        data_dict = {
            'covariance': cov[:, indices].detach().cpu(),
            'adapted_cov': inv_cov[:, indices].detach().cpu(),
            'adaptive_scales': scales[:, indices].detach().cpu(),
            'eigenvectors': eigenvectors[:, indices].detach().cpu(),
            'anchor_world': xyz[:, indices].detach().cpu(),
        }

        # Save the data
        scene_name, file_name = self.curr_name.split('/')[0], self.curr_name.split('/')[1]
        save_pcd = Path(self.save_da + f'_{name}') / scene_name
        save_pcd.mkdir(exist_ok=True, parents=True)
        save_path = str(save_pcd / f'cov_{file_name}.pt')
        torch.save(data_dict, save_path)
        print(f"save cov_{file_name}.pt to {save_pcd}, size: {cov.shape}")

    def convert_da_refine_color(self, mean, semantics, name='cam', ignore=None):
        # mean: (num, 3), semantics: (num, C)
        if isinstance(mean, torch.Tensor):
            pts = mean.detach().cpu().numpy()  # (num, 3)
        else:
            pts = copy.deepcopy(mean)  # (num, 3)

        sems = semantics.argmax(dim=1).cpu().numpy()  # (num, 1)
        colors = np.array([color_list[sem+1] for sem in sems])  # (num, 3)

        if isinstance(ignore, list):
            mask = np.ones_like(sems, dtype=bool)  # (num, 1)
            for c in ignore:
                mask &= (sems != c)
        else:
            if ignore is not None:
                mask = (sems != ignore)
            else:
                mask = np.ones_like(sems, dtype=bool)  # (num, 1)

        print(f"pts: {pts.shape}, colors: {colors.shape}")
        pts = pts[mask]
        colors = colors[mask]
        print(f"pts: {pts.shape}, colors: {colors.shape}")

        # Generate PCD
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        scene_name, file_name = self.curr_name.split('/')[0], self.curr_name.split('/')[1]
        save_pcd = Path(self.save_da + f'_{name}') / scene_name
        save_pcd.mkdir(exist_ok=True, parents=True)
        save_path = str(save_pcd / f'pcd_{file_name}.ply')
        o3d.io.write_point_cloud(save_path, pcd)
        print(f"save pcd_{file_name}.ply to {save_pcd}, size: {pts.shape}")

    def convert_gaussian_ellipsoids(self, gaussian, semantics, name='cam', alpha=1.0, ignore=None):
        # semantics: (num, C), mask_gaussian: (num)
        xyz = gaussian.means.squeeze(0).detach().cpu().numpy()  # [anchor num, 3]
        scale = gaussian.scales.squeeze(0).detach().cpu().numpy()  # [anchor num, 3]
        rotation = gaussian.rotations.squeeze(0).detach().cpu().numpy()  # [anchor num, 4]
        opacities = gaussian.opacities.squeeze(0).detach().cpu().numpy()  # [anchor num, 1]

        sems = semantics.argmax(dim=1).detach().cpu().numpy()  # (num, 1)
        colors = np.array([color_list[sem+1] for sem in sems])  # (num, 3)
        shs = colors * sqrt(4 * pi)  # (num, 3)

        flat_semantic = sems.flatten()
        unique_sem, counts_sem = np.unique(flat_semantic, return_counts=True)
        sem_counts = dict(zip(unique_sem, counts_sem))
        print("semantic counts as a dictionary:", sem_counts)

        if isinstance(ignore, list):
            mask = np.ones_like(sems, dtype=bool)  # (num, 1)
            for c in ignore:
                mask &= (sems != c)
        else:
            if ignore is not None:
                mask = (sems != ignore)
            else:
                mask = np.ones_like(sems, dtype=bool)  # (num, 1)

        xyz = xyz[mask]  # [anchor num, 3]
        scale = scale[mask]  # [anchor num, 3]
        rotation = rotation[mask]  # [anchor num, 4]
        opacities = opacities[mask]  # [anchor num, 1]
        shs = shs[mask] * alpha  # [anchor num, 3]

        scale = np.log(scale)
        opacities = np.log(opacities/(1-opacities))

        dtype_full = [(attribute, 'f4') for attribute in gaussian_attributes]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, opacities, shs, scale, rotation), axis=-1)
        elements[:] = list(map(tuple, attributes))

        # file path
        scene_name, file_name = self.curr_name.split('/')[0], self.curr_name.split('/')[1]
        save_pcd = Path(self.save_da + f'_{name}') / scene_name
        save_pcd.mkdir(exist_ok=True, parents=True)
        save_path = str(save_pcd / f'pcd_{file_name}.ply')

        el = PlyElement.describe(elements, 'vertex')
        ply = PlyData([el])
        ply.write(save_path)

        print(f"Successfully Save Gaussian Ellipsoids to {save_path}, Shape: {attributes.shape}")


class OccProbDebug:
    """
    A unified class for occupancy grid generation and loss computation.
    Supports both geometric and semantic-geometric loss calculations.
    """

    def __init__(
            self,
            grid_shape: Tuple[int, int, int] = (60, 60, 36),
            grid_size: float = 0.08,
            epsilon: float = 1e-6,
            max_clamp: float = 15.0,
            loss_type: str = 'geo_3',
            cuda_kwargs: Optional[Dict] = None,
    ):
        """
        Initialize the occupancy grid processor.

        Args:
            grid_shape: Shape of the occupancy grid (x, y, z)
            grid_size: Size of each grid cell
            epsilon: Small value to prevent division by zero
            max_clamp: Maximum value for loss clamping
            cuda_kwargs: CUDA aggregator parameters
        """
        self.grid_shape = grid_shape
        self.grid_size = grid_size
        self.epsilon = epsilon
        self.max_clamp = max_clamp
        self.cuda_kwargs = cuda_kwargs
        loss_str = loss_type.split('_')  # eg, 'sem_3' --> ['sem', '3']
        self.sem_or_geo = loss_str[0]  # 'sem' or 'geo'

        # Configure which losses to compute
        self.loss_config = {
            'loss_precision': True,
            'loss_recall': True,
            'loss_spec': True,
            'loss_bce': True,
            'loss_dice': True,
        }

        # Precomputed offset dictionary for different radii
        self.offset_dict = {
            '0.08': [-1, 0, 1], '0.10': [-1, 0, 1], '0.12': [-1, 0, 1],
            '0.16': [-2, -1, 0, 1], '0.20': [-2, -1, 0, 1], '0.24': [-2, -1, 0, 1],
            '0.32': None, '0.36': None, '0.40': None
        }

        # Colors for output formatting
        self.colors = {
            'BOLD': '\033[1m',
            'CYAN': '\033[96m',
            'END': '\033[0m'
        }

        self.class_dict = {
            "1": "Ceiling",
            "2": "Floor",
            "3": "Wall",
            "4": "Window",
            "5": "Chair",
            "6": "Bed",
            "7": "Sofa",
            "8": "Table",
            "9": "TVs",
            "10": "Furniture",
            "11": "Object",
            "12": "Empty"
        }

        # Initialize CUDA aggregator if provided
        self.occ_aggregator = None
        if cuda_kwargs is not None:
            try:
                from loss.ops.occ_prob import OccAggregator
                self.occ_aggregator = OccAggregator(**cuda_kwargs)
                print("CUDA aggregator initialized successfully!")
            except ImportError:
                print("Warning: CUDA aggregator not available, falling back to PyTorch implementation")

    def xyz_to_occupancy(
            self,
            anchors_xyz: torch.Tensor,
            origin_use: torch.Tensor,
            off_range: Optional[List[int]] = None,
            radius: float = 0.08,
    ) -> torch.Tensor:
        """
        Convert 3D anchor points to occupancy grid.

        Args:
            anchors_xyz: Anchor points [num_anchors, 3]
            origin_use: Grid origin [3]
            off_range: Neighborhood offset range
            radius: Influence radius

        Returns:
            Occupancy grid [1, H, W, D]
        """
        num_anchor = anchors_xyz.shape[0]
        device = anchors_xyz.device

        if num_anchor == 0:
            return torch.zeros((1, *self.grid_shape), device=device, dtype=torch.float32)

        # Convert to grid indices
        xyz_world_int = ((anchors_xyz - origin_use) / self.grid_size).detach().to(torch.long)

        if off_range is None:
            # Use all grid indices - optimized version
            total_voxels = np.prod(self.grid_shape)
            z_coords = torch.arange(self.grid_shape[2], device=device, dtype=torch.long)
            y_coords = torch.arange(self.grid_shape[1], device=device, dtype=torch.long)
            x_coords = torch.arange(self.grid_shape[0], device=device, dtype=torch.long)

            # More efficient meshgrid creation
            z_flat = z_coords.repeat(self.grid_shape[0] * self.grid_shape[1])
            y_flat = y_coords.repeat_interleave(self.grid_shape[2]).repeat(self.grid_shape[0])
            x_flat = x_coords.repeat_interleave(self.grid_shape[1] * self.grid_shape[2])

            valid_indices = torch.stack([x_flat, y_flat, z_flat], dim=1)
        else:
            # Generate neighborhood offsets
            offsets = torch.tensor([
                [i, j, k] for i in off_range for j in off_range for k in off_range
            ], device=device, dtype=torch.long)

            # Vectorized neighbor computation
            neighbor_indices = xyz_world_int.unsqueeze(1) + offsets.unsqueeze(0)
            neighbor_indices = neighbor_indices.view(-1, 3)

            # Efficient bounds checking
            valid_mask = torch.all(
                (neighbor_indices >= 0) &
                (neighbor_indices < torch.tensor(self.grid_shape, device=device)),
                dim=1
            )
            valid_indices = neighbor_indices[valid_mask]

        # Remove duplicates efficiently
        if valid_indices.numel() > 0:
            valid_indices_unique = torch.unique(valid_indices, dim=0)

            # Convert to world coordinates
            valid_voxels = valid_indices_unique.float() * self.grid_size + origin_use + self.grid_size / 2

            # Efficient distance computation
            dist_sq = torch.cdist(valid_voxels, anchors_xyz, p=2).pow(2)
            influence_radius_sq = radius ** 2
            influence_scores = torch.exp(-dist_sq / (influence_radius_sq + self.epsilon))
            aggregated_influence = torch.sum(influence_scores, dim=1)
            occupancy_probs = 1.0 - torch.exp(-aggregated_influence)

            # Efficient scatter operation
            linear_indices = (valid_indices_unique[:, 0] * self.grid_shape[1] * self.grid_shape[2] +
                              valid_indices_unique[:, 1] * self.grid_shape[2] +
                              valid_indices_unique[:, 2])

            occupancy_flat = torch.zeros(np.prod(self.grid_shape), device=device, dtype=torch.float32)
            occupancy_flat.scatter_(0, linear_indices, occupancy_probs)

            return occupancy_flat.reshape(1, *self.grid_shape)
        else:
            return torch.zeros((1, *self.grid_shape), device=device, dtype=torch.float32)

    def _compute_base_metrics(self, valid_target: torch.Tensor, valid_probs: torch.Tensor) -> Dict:
        """
        Compute base metrics (precision, recall, specificity) and additional loss functions.

        Args:
            valid_target: Ground truth binary labels
            valid_probs: Predicted probabilities

        Returns:
            Dictionary containing metrics and additional losses
        """
        result = {}

        # Basic components
        spec_probs = 1 - valid_probs
        spec_target = 1 - valid_target
        intersection = (valid_target * valid_probs).sum()

        # Calculate base metrics (same as original logic)
        precision = intersection / (valid_probs.sum() + self.epsilon)
        recall = intersection / (valid_target.sum() + self.epsilon)
        specificity = (spec_target * spec_probs).sum() / (spec_target.sum() + self.epsilon)

        result['precision'] = precision.item()
        result['recall'] = recall.item()
        result['specificity'] = specificity.item()

        # Additional loss functions (new)
        if self.loss_config.get('loss_bce', False):
            result['loss_bce'] = F.binary_cross_entropy(valid_probs, valid_target, reduction='mean').item()

        if self.loss_config.get('loss_dice', False):
            denominator = valid_probs.sum() + valid_target.sum()
            dice_coeff = (2.0 * intersection + self.epsilon) / (denominator + self.epsilon)
            result['loss_dice'] = (1.0 - dice_coeff).item()

        return result

    def compute_geometric_loss(
            self,
            ssc_bin: torch.Tensor,
            gaussian,
            cov_inv: torch.Tensor,
            sampled_xyz: torch.Tensor,
            pc_min: torch.Tensor,
            off_range: Optional[List[int]],
            radius: float,
            mask: torch.Tensor
    ) -> Dict:
        """
        Compute geometric loss for occupancy prediction.
        """
        # Generate occupancy probability
        if self.occ_aggregator is not None:
            means = gaussian.means
            scales = gaussian.scales
            opacities = gaussian.opacities.flatten(1, 2)
            occ_prob = self.occ_aggregator(sampled_xyz, means, scales, cov_inv, opacities, pc_min)
            occ_prob = occ_prob.reshape(*self.grid_shape).unsqueeze(0)
        else:
            loc_xyz = gaussian.means.float()
            occ_prob = self.xyz_to_occupancy(loc_xyz, pc_min, off_range, radius)

        occ_prob = torch.clamp(occ_prob, self.epsilon, 1 - self.epsilon)

        # Get base metrics and additional losses
        valid_target = ssc_bin[mask].float()
        valid_probs = occ_prob[mask]
        spec_probs = (1 - occ_prob)[mask]

        intersection = (valid_target * valid_probs).sum()
        precision = intersection / (valid_probs.sum() + self.epsilon)
        recall = intersection / (valid_target.sum() + self.epsilon)

        spec_target = 1 - valid_target
        specificity = (spec_target * spec_probs).sum() / (spec_target.sum() + self.epsilon)

        # Compute losses (original logic)
        loss_precision = F.binary_cross_entropy(precision, torch.ones_like(precision))
        loss_recall = F.binary_cross_entropy(recall, torch.ones_like(recall))
        loss_spec = F.binary_cross_entropy(specificity, torch.ones_like(specificity))
        loss = loss_precision + loss_recall + loss_spec

        result = {
            'loss': loss.item(),
            'loss_precision': loss_precision.item(),
            'loss_recall': loss_recall.item(),
            'loss_spec': loss_spec.item(),
            'occ_prob': occ_prob,
            'precision': precision.item(),
            'recall': recall.item(),
            'specificity': specificity.item()
        }

        # Add additional losses if configured
        base_metrics = self._compute_base_metrics(valid_target, valid_probs)
        for key in ['loss_bce', 'loss_dice']:
            if key in base_metrics:
                result[key] = base_metrics[key]

        return result

    def compute_semantic_geometric_loss(
            self,
            ssc_target: torch.Tensor,
            gaussian,
            cov_inv: torch.Tensor,
            sampled_xyz: torch.Tensor,
            pc_min: torch.Tensor,
            sem_cls_range: List[int],
            ignore_label: int,
            empty_idx: int,
            mask: torch.Tensor,
            semantic=None,
    ) -> Dict:
        """
        Compute semantic-geometric loss for multi-class occupancy prediction.
        """
        result = {
            'loss': [],
            'loss_precision': [],
            'loss_recall': [],
            'loss_spec': [],
            'occ_prob': [],
            'cls': [],
        }

        # Add additional loss types to result structure if configured
        if self.loss_config.get('loss_bce', False):
            result['loss_bce'] = []
        if self.loss_config.get('loss_dice', False):
            result['loss_dice'] = []

        cls_indices = torch.argmax(gaussian.semantics.squeeze(0), dim=-1)
        unique_indices = torch.unique(cls_indices)
        total_loss = 0.0
        valid_count = 0

        print(f"unique_indices: {unique_indices}")

        for class_id in range(sem_cls_range[0], sem_cls_range[1]):
            if class_id in [ignore_label, empty_idx]:
                continue

            # Check if class exists in GT and predictions
            target = ssc_target[mask]
            completion_target = torch.ones_like(target)
            completion_target[target != class_id] = 0

            print(f"idx: {class_id}, name: {self.class_dict[f'{class_id}']}, "
                  f"label valid: {torch.sum(completion_target) > 0}, pred valid: {class_id in unique_indices}")

            if (torch.sum(completion_target) > 0) or (class_id in unique_indices):

                if 'geo' in self.sem_or_geo:
                    loc_gaussian_mask = torch.ones_like(cls_indices)
                    loc_gaussian_mask[cls_indices != (class_id - 1)] = 0

                    if not (torch.sum(cls_indices) > 0):
                        continue

                    # Generate occupancy probability for this class
                    if torch.sum(loc_gaussian_mask) > 0:
                        if self.occ_aggregator is not None:
                            loc_means = gaussian.means[:, loc_gaussian_mask]
                            loc_scales = gaussian.scales[:, loc_gaussian_mask]
                            loc_cov_inv = cov_inv[:, loc_gaussian_mask]
                            opacities = gaussian.opacities.flatten(1, 2)
                            loc_opas = opacities[:, loc_gaussian_mask]

                            occ_prob = self.occ_aggregator(sampled_xyz, loc_means, loc_scales,
                                                           loc_cov_inv, loc_opas, pc_min)
                            occ_prob = occ_prob.reshape(*self.grid_shape).unsqueeze(0)
                        else:
                            pred_probs = gaussian.means.float()
                            occ_prob = self.xyz_to_occupancy(pred_probs.squeeze(0), pc_min)

                        occ_prob = torch.clamp(occ_prob, self.epsilon, 1 - self.epsilon)
                    else:
                        occ_prob = torch.zeros_like(ssc_target, dtype=torch.float32)
                    valid_probs = occ_prob[mask]
                else:
                    occ_prob = semantic[:, class_id]
                    valid_probs = occ_prob[mask]

                valid_count += 1

                # Compute metrics (keeping original logic)
                valid_target = completion_target.float()
                nominator = (valid_target * valid_probs).sum()

                # recall
                recall = nominator / (valid_target.sum() + self.epsilon)
                loc_loss_recall = F.binary_cross_entropy(recall, torch.ones_like(recall))
                loc_loss_recall = torch.clamp(loc_loss_recall, max=self.max_clamp)
                loss_class = loc_loss_recall
                result['loss_recall'].append(loc_loss_recall.item())

                # precision
                if torch.sum(valid_probs) > 0:
                    precision = nominator / (valid_probs.sum() + self.epsilon)
                    loc_loss_precision = F.binary_cross_entropy(precision, torch.ones_like(precision))
                    loc_loss_precision = torch.clamp(loc_loss_precision, max=self.max_clamp)
                    loss_class += loc_loss_precision
                    result['loss_precision'].append(loc_loss_precision.item())
                else:
                    result['loss_precision'].append(0.0)

                # specificity
                if torch.sum(1 - completion_target) > 0:
                    specificity = (((1 - valid_target) * (1 - valid_probs)).sum() /
                                   ((1 - valid_target).sum() + self.epsilon))
                    loss_spec = F.binary_cross_entropy(specificity, torch.ones_like(specificity))
                    loss_spec = torch.clamp(loss_spec, max=self.max_clamp)
                    loss_class += loss_spec
                    result['loss_spec'].append(loss_spec.item())
                else:
                    result['loss_spec'].append(0.0)

                total_loss += loss_class

                # Add additional losses if configured
                if self.loss_config.get('loss_bce', False):
                    loss_bce = F.binary_cross_entropy(valid_probs, valid_target, reduction='mean')
                    result['loss_bce'].append(loss_bce.item())

                denominator = valid_probs.sum() + valid_target.sum()
                dice_coeff = (2.0 * nominator + self.epsilon) / (denominator + self.epsilon)
                loss_dice = 1.0 - dice_coeff
                result['loss_dice'].append(loss_dice.item())

                # Store basic results
                result['loss'].append(total_loss.item())
                result['occ_prob'].append(occ_prob)
                result['cls'].append(class_id)

        result['total_loss'] = total_loss / valid_count if valid_count > 0 else torch.tensor(0.0)
        return result

    def compute_probability_scale_loss(
            self,
            output_dict: Dict,
            nyu_pc_min: torch.Tensor,
            sampled_xyz: torch.Tensor,
            convert_list: List[int] = [0, 1, 2],
            radii: List[float] = [0.16],
            convert_data=None,
            log_view: bool = False,
    ) -> pd.DataFrame:
        """
        Main function to compute probability scale loss across multiple layers and radii.
        """
        # Prepare data
        self.convert_data = convert_data
        ssc_target = output_dict['ce_label'].long()
        ssc_bin = ((ssc_target != 0) & (ssc_target != 12)).to(torch.uint8)
        mask = output_dict['fov_mask'].unsqueeze(0)

        all_results = []

        # Process each radius
        for radius in radii:
            off_range = self.offset_dict.get(f'{radius:.2f}')

            # Process each layer
            for layer_idx in convert_list:
                start_time = time.time()

                # Compute loss based on type
                if self.sem_or_geo == 'geo':
                    loss_result = self.compute_geometric_loss(
                        ssc_bin, output_dict['gaussian_cache'][layer_idx],
                        output_dict['cov_inv_cache'][layer_idx],
                        sampled_xyz, nyu_pc_min, off_range, radius, mask
                    )

                    # Handle visualization
                    if log_view and self.convert_data is not None:
                        self._visualize_results(loss_result['occ_prob'], sampled_xyz,
                                                layer_idx, 0, 'geo')

                    # Store results - dynamically extract loss components
                    result_entry = {
                        'radius': radius,
                        'time': time.time() - start_time,
                        'layer': layer_idx,
                        'valid_cls': 0,
                    }

                    # Add all loss components dynamically
                    for key, value in loss_result.items():
                        if key.startswith('loss') or key in ['precision', 'recall', 'specificity']:
                            if isinstance(value, (int, float)):
                                # Remove 'loss_' prefix for display
                                display_key = key.replace('loss_', '') if key.startswith('loss_') else key
                                result_entry[display_key] = value

                    all_results.append(result_entry)

                elif 'sem' in self.sem_or_geo:
                    loss_result = self.compute_semantic_geometric_loss(
                        ssc_target, output_dict['gaussian_cache'][layer_idx],
                        output_dict['cov_inv_cache'][layer_idx],
                        sampled_xyz, nyu_pc_min, [1, 12], 0, 12, mask,
                        output_dict['sem_cache'][layer_idx],
                    )

                    # Handle visualization and results for each class
                    for i, (cls_id, occ_prob) in enumerate(zip(loss_result['cls'], loss_result['occ_prob'])):
                        if log_view and self.convert_data is not None:
                            self._visualize_results(occ_prob, sampled_xyz, layer_idx, cls_id, 'sem-geo')

                        result_entry = {
                            'radius': radius,
                            'time': time.time() - start_time,
                            'layer': layer_idx,
                            'valid_cls': cls_id,
                        }

                        # Add all loss components dynamically
                        for key in loss_result:
                            if key.startswith('loss') and isinstance(loss_result[key], list):
                                if i < len(loss_result[key]):
                                    # Remove 'loss_' prefix for display
                                    display_key = key.replace('loss_', '') if key.startswith('loss_') else key
                                    result_entry[display_key] = loss_result[key][i]

                        all_results.append(result_entry)

        # Create and display results
        df = pd.DataFrame(all_results)
        self._display_results(df, convert_list)
        return df

    def _visualize_results(self, occ_prob: torch.Tensor, sampled_xyz: torch.Tensor,
                           layer_idx: int, class_id: int, loss_type: str):
        """Helper function for visualization"""
        if self.convert_data is None:
            return

        loc_occ_prob = occ_prob.flatten()
        loc_mask = (loc_occ_prob > 0.3).squeeze()

        if loc_mask.sum() > 0:
            loc_voxel_centers = sampled_xyz.squeeze(0)[loc_mask]
            cmap = plt.cm.coolwarm
            colors = cmap(loc_occ_prob.cpu().numpy().squeeze())[:, :3]
            colors = colors[loc_mask.cpu().numpy()]
            self.convert_data.convert_da_pts_color(loc_voxel_centers, colors,
                                                   f'refine_{layer_idx}', class_id)

    def _display_results(self, df: pd.DataFrame, convert_list: List[int]):
        """Display formatted results table with dynamic columns"""
        if df.empty:
            print("No results to display")
            return

        # Get dynamic columns (exclude non-numeric display columns)
        base_cols = ['radius', 'time', 'layer', 'valid_cls']
        metric_cols = [col for col in df.columns if col not in base_cols]
        all_cols = base_cols + sorted(metric_cols)

        # Calculate column widths dynamically
        col_widths = {}
        for col in all_cols:
            if col in base_cols:
                if col == 'radius':
                    col_widths[col] = 8
                elif col == 'time':
                    col_widths[col] = 8
                elif col == 'layer':
                    col_widths[col] = 6
                elif col == 'valid_cls':
                    col_widths[col] = 10
            else:
                # For metric columns, calculate based on data
                max_width = len(col) + 2
                if col in df.columns:
                    data_widths = [len(f"{val:.4f}") for val in df[col] if pd.notna(val)]
                    if data_widths:
                        max_width = max(max_width, max(data_widths) + 2)
                col_widths[col] = max_width

        # Print header
        header = ''.join([f"{col:<{col_widths[col]}}" for col in all_cols])
        print(header)
        print("-" * len(header))

        # Find best loss for each layer
        layer_best_indices = {}
        if 'loss' in df.columns:
            for layer_idx in convert_list:
                layer_data = df[df['layer'] == layer_idx]
                if not layer_data.empty:
                    best_idx = layer_data['loss'].idxmin()
                    layer_best_indices[layer_idx] = best_idx

        # Print rows with highlighting
        current_layer = None
        for i, row in df.iterrows():
            if current_layer is not None and row['layer'] != current_layer:
                print("-" * len(header))
            current_layer = row['layer']

            is_best = row['layer'] in layer_best_indices and layer_best_indices[row['layer']] == i

            row_parts = []
            for col in all_cols:
                if col in ['radius', 'time'] and col in row:
                    row_parts.append(f"{row[col]:.4f}".ljust(col_widths[col]))
                elif col in ['layer', 'valid_cls'] and col in row:
                    row_parts.append(f"{int(row[col])}".ljust(col_widths[col]))
                elif col in row:
                    # Handle metric columns
                    value_str = f"{row[col]:.4f}" if pd.notna(row[col]) else "N/A"

                    # Highlight best loss
                    if col == 'loss' and is_best:
                        formatted = f"{self.colors['BOLD']}{self.colors['CYAN']}{value_str}{self.colors['END']}"
                        padding = col_widths[col] - len(value_str)
                        formatted = formatted + ' ' * padding
                    else:
                        formatted = value_str.ljust(col_widths[col])

                    row_parts.append(formatted)
                else:
                    row_parts.append(' ' * col_widths[col])

            print(''.join(row_parts))


"""
# label colors
color_list = [
    [0.0, 0.0, 0.0],  # 0: Background or Unlabeled
    [0.7, 0.4, 0.4],  # 1: Ceiling – Muted Red -- 0
    [0.4, 0.7, 0.4],  # 2: Floor – Muted Green -- 1
    [0.4, 0.4, 0.7],  # 3: Wall – Muted Blue -- 2
    [0.7, 0.7, 0.4],  # 4: Window – Muted Yellow -- 3
    [0.7, 0.4, 0.7],  # 5: Chair – Muted Magenta -- 4
    [0.4, 0.7, 0.7],  # 6: Bed – Muted Cyan -- 5
    [0.6, 0.6, 0.6],  # 7: Sofa – Muted Gray -- 6
    [0.8, 0.6, 0.4],  # 8: Table – Muted Orange -- 7
    [0.5, 0.6, 0.4],  # 9: TVs – Muted Olive -- 8
    [0.4, 0.6, 0.6],  # 10: Furniture – Muted Teal -- 9
    [0.4, 0.5, 0.8],  # 11: Object – Muted Blue Variant -- 10
    [0.6, 0.4, 0.6],  # 12: (Empty) – Muted Purple -- 11
    # [0.0, 0.0, 0.0],  # 255: Background or Unlabeled
]
"""
