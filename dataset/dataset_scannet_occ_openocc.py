import os
import time
import json
import glob
import numpy as np
import numba as nb
import torch
from sympy.codegen.cnodes import static
from torch.utils import data
import pickle
from PIL import Image
from numba import njit, prange
from mmcv.image.io import imread
import copy
from pyquaternion import Quaternion

import math, cv2
from . import OPENOCC_DATASET
from dataset.nyu_utils import vox2pix, custom_vox2pix
from torchvision import transforms
from mmcv.image.io import imread
from torchvision.transforms import Compose
from dataset.transform_ import Resize, NormalizeImage, PrepareForNet
from model.segmentor.gaussian_segmentor.utils import rigid_transform, transform_vox_range, occ2world, ConvertData, color_list
# from vggt.utils.load_fn import load_and_preprocess_images


@OPENOCC_DATASET.register_module()
class Scannet_Scene_OpenOccupancy_Dataset(data.Dataset):
    def __init__(
        self,
        data_path, 
        num_frames=1,
        offset=0,
        grid_size_occ=[60, 60, 36],
        coarse_ratio=2,
        empty_idx=0,
        phase='train',
        num_pts=21600,
        data_tg='base',
        data_sorted = False,
        stage_tg='main',
        ):

        self.occscannet_root = data_path
        self.data_sorted = data_sorted
        self.stage_tg = stage_tg
        self.phase = phase
        
        self.num_frames = num_frames
        self.offset = offset
        self.grid_size_occ = grid_size_occ
        self.grid_size_occ_coarse = (np.array(grid_size_occ) // coarse_ratio).astype(np.uint32)
        self.coarse_ratio = coarse_ratio
        self.empty_idx = empty_idx
        self.phase = phase

        self.voxel_size = 0.08  # 0.08m
        self.scene_size = (4.8, 4.8, 2.88)  # (4.8m, 4.8m, 2.88m)
        if data_tg == 'base':
            if data_sorted:
                subscenes_list = f'{self.occscannet_root}/{self.phase}_final_sorted.txt'
            else:
                subscenes_list = f'{self.occscannet_root}/{self.phase}_final.txt'
        elif data_tg == 'mini':  # 70 elements in same scene
            if data_sorted:
                subscenes_list = f'{self.occscannet_root}/{self.phase}_mini_final_sorted.txt'
            else:
                subscenes_list = f'{self.occscannet_root}/{self.phase}_mini_final.txt'
        with open(subscenes_list, 'r') as f:
            self.used_subscenes = f.readlines()
            for i in range(len(self.used_subscenes)):
                self.used_subscenes[i] = f'{self.occscannet_root}/' + self.used_subscenes[i].strip()
        
        self.num_pts = num_pts
        
        self.normalize_rgb = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return len(self.used_subscenes)

    def __getitem__(self, index):
        name = self.used_subscenes[index]
        with open(name, 'rb') as f:
            data = pickle.load(f)

        name_without_ext = os.path.splitext(name)[0]
        this_name = name_without_ext.split('gathered_data/')[-1]

        self.base_time_toc_0 = time.time()

        meta = {}
        meta['name'] = this_name # 'scene0000_00/00000'
        meta['scene_size'] = self.scene_size
        cam_pose = data['cam_pose']
        meta['cam2world'] = cam_pose
        world2cam = np.linalg.inv(cam_pose)
        meta['world2cam'] = world2cam

        save_root = f'{self.occscannet_root}/depth/vis_occ'
        convert_data = ConvertData(
            this_name,
            save_root,
        )

        rgb_path = f'{self.occscannet_root}/posed_images/' + f'{this_name}.jpg'
        depth_path = f'{self.occscannet_root}/posed_images/' + f'{this_name}.png'
        depth_gt_np = Image.open(depth_path).convert('I;16')
        depth_gt_np = np.array(depth_gt_np) / 1000.0

        transform = Compose([
            Resize(
                width=480,
                height=480,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

        self.base_time_toc_1 = time.time()

        img_depthbranch = cv2.imread(rgb_path)
        img_depthbranch = cv2.resize(img_depthbranch, (640, 480), interpolation=cv2.INTER_NEAREST)
        img_depthbranch = cv2.cvtColor(img_depthbranch, cv2.COLOR_BGR2RGB) / 255.0
        sample = transform({'image': img_depthbranch, 'depth': depth_gt_np})
        img_depthbranch = torch.from_numpy(sample['image']).unsqueeze(0)
        depth_gt_np = torch.from_numpy(sample['depth']).unsqueeze(0)
        meta['depth_gt_np'] = depth_gt_np  # (1, 480, 640)
        # convert_data.convert_depth_np(depth_gt_np.squeeze(0), 'depth_gt_np')
        depth_valid_mask = (torch.isnan(depth_gt_np) == 0)
        depth_gt_np[depth_valid_mask == 0] = 0
        meta['img_depthbranch'] = img_depthbranch
        meta['depth_gt_np_valid'] = depth_gt_np

        meta['rgb_path'] = rgb_path
        N_img = []
        this_img = imread(rgb_path, 'unchanged').astype(np.float32)
        this_H, this_W, _ = this_img.shape
        new_H, new_W = 480, 640
        # resize
        new_img = cv2.resize(this_img, (new_W, new_H))
        W_factor = new_W / this_W
        H_factor = new_H / this_H
        N_img.append(new_img)
        img = np.stack(N_img, 0)  # [1, 968, 1296, 3]
        this_H, this_W= new_H, new_W
        img = [img]  # [1, 1, 968, 1296, 3]
        meta['rgb'] = img  # [1, 1, 968, 1296, 3]

        self.base_time_toc_2 = time.time()

        # img_vggt = load_and_preprocess_images([rgb_path])
        # meta['img_vggt'] = img_vggt  # (1, 3, w, h)

        self.vggt_time_toc = time.time()

        cam_intrin = data['intrinsic']
        cam_intrin[0, 0] *= W_factor
        cam_intrin[0, 2] *= W_factor
        cam_intrin[1, 1] *= H_factor
        cam_intrin[1, 2] *= H_factor

        meta['cam_k'] = cam_intrin[:3, :3]
        viewpad = np.eye(4)
        viewpad[:meta['cam_k'].shape[0], :meta['cam_k'].shape[1]] = meta['cam_k']
        meta['cam2img'] = viewpad
        world2img = (viewpad @ world2cam)
        meta['world2img'] = world2img

        meta['depth_path'] = depth_path
        depth_gt = Image.open(depth_path).convert('I;16')
        depth_gt = np.array(depth_gt) / 1000.0
        meta['depth_gt'] = depth_gt  # [480, 640]
        meta['valid_depth_mask'] = depth_gt > 0  # [480, 640]
        # convert_data.convert_depth(depth_gt, 'depth_gt')

        self.base_time_toc_3 = time.time()

        if self.stage_tg == 'depth_branch':
            pts_world, pts_cam, valid_pts_mask = self.depth2world(depth_gt, cam_intrin[:3, :3], cam_pose)
            meta['valid_pts_mask'] = valid_pts_mask
            meta['pts_world_gt'] = pts_world
            # meta['pts_cam_gt'] = pts_cam
            # convert_data.convert_da_pts_nocolor(pts_cam, 'gt_pts_cam')
            # convert_data.convert_da_pts_nocolor(pts_world, 'gt_pts_world')

        self.depth_time_toc = time.time()

        vox_origin = data["voxel_origin"]
        meta['vox_origin'] = np.round(np.array(vox_origin, dtype=np.float32), 4)
        target = data["target_1_4"] # 60, 60, 36
        target = np.transpose(target, (1, 0, 2))
        # replace unknown '255' with '0'，replace empty '0' with '12'
        target[target == 0] = 12
        target[target == 255] = 0 
        occ = target # (60, 60, 36)
        nonemptymask = (occ != 12)
        nonignoremask = (occ != 0)
        occ = [occ] # [1, 60, 60, 36]

        # compute the 3D-2D mapping
        projected_pix, fov_mask, pix_z, occ_xyz, cam_pts = custom_vox2pix(
            world2cam,
            meta['cam_k'],
            meta['vox_origin'],
            self.voxel_size,
            this_W,
            this_H,
            self.scene_size,
            dim_60_60_36=True,
        )
        # occ_cam = rigid_transform(occ_xyz, world2cam)

        # convert_data.convert_da_pts_nocolor(occ_xyz, 'world')
        # convert_data.convert_da_pts_nocolor(cam_pts, 'cam')

        _, fov_mask_4, _, _ = vox2pix(
            world2cam,
            meta['cam_k'],
            meta['vox_origin'],
            self.voxel_size * 4,
            this_W,
            this_H,
            self.scene_size,
            dim_60_60_36=False,
        )
        meta['projected_pix'] = projected_pix
        meta['fov_mask'] = fov_mask.reshape(60, 60, 36)
        # meta['fov_mask_4'] = fov_mask_4.reshape(15, 15, 9)

        meta['pix_z'] = pix_z
        meta['occ_xyz'] = occ_xyz.reshape(60, 60, 36, 3)
        # meta['occ_cam'] = occ_cam.reshape(60, 60, 36, 3)

        vox_near = meta['vox_origin']
        vox_far = vox_near + meta['scene_size']
        nyu_pc_range = np.concatenate([vox_near, vox_far], axis=0)
        meta['nyu_pc_range'] = nyu_pc_range.astype(np.float32)
        cam_pc_range = transform_vox_range(nyu_pc_range, world2cam).astype(np.float32)
        meta['cam_pc_range'] = cam_pc_range

        # scan = meta['occ_xyz'][nonemptymask]
        # meta['occ_xyz_nonempty'] = scan
        # meta['num_depth'] = self.num_pts
        # if scan.shape[0] < self.num_pts:
        #     multi = int(math.ceil(self.num_pts * 1.0 / scan.shape[0])) - 1
        #     scan_ = np.repeat(scan, multi, 0)
        #     scan_ = scan_ + np.random.randn(*scan_.shape) * 0.01
        #     scan_ = scan_[np.random.choice(scan_.shape[0], self.num_pts - scan.shape[0], False)]
        #     scan_[:, 0] = np.clip(scan_[:, 0], nyu_pc_range[0], nyu_pc_range[3])
        #     scan_[:, 1] = np.clip(scan_[:, 1], nyu_pc_range[1], nyu_pc_range[4])
        #     scan_[:, 2] = np.clip(scan_[:, 2], nyu_pc_range[2], nyu_pc_range[5])
        #     scan = np.concatenate([scan, scan_], 0)
        # else:
        #     scan = scan[np.random.choice(scan.shape[0], self.num_pts, False)]

        # scan[:, 0] = (scan[:, 0] - nyu_pc_range[0]) / (nyu_pc_range[3] - nyu_pc_range[0])
        # scan[:, 1] = (scan[:, 1] - nyu_pc_range[1]) / (nyu_pc_range[4] - nyu_pc_range[1])
        # scan[:, 2] = (scan[:, 2] - nyu_pc_range[2]) / (nyu_pc_range[5] - nyu_pc_range[2])

        # meta['anchor_points'] = scan
        # convert_data.convert_da_pts_nocolor(scan, 'anchor_points')

        cam_vox_near = np.array([-5, -6, -3])
        cam_vox_far = np.array([5, 6, 8])
        nyu_vox_range = np.concatenate([cam_vox_near, cam_vox_far], axis=0).astype(np.float32)
        meta['nyu_vox_range'] = nyu_vox_range
        meta['cam_vox_range'] = nyu_vox_range

        # lut_range = torch.tensor([-10.24, -10.24, -5.0, 10.24, 10.24, 3.0], dtype=torch.float32)
        # meta['lut_range'] = lut_range

        # meta['occ_mask_valid'] = (occ != 0)
        # meta['occ_mask_valid_fov'] = (occ != 0) & fov_mask
        meta['label'] = occ

        imgs = np.stack(img, 0)
        occs = np.stack(occ, 0)

        # occ_world, color_world = occ2world(occs, meta['vox_origin'], self.voxel_size)  # [num, 3]
        # occ_cam = rigid_transform(occ_world, world2cam)  # [num, 3]

        # meta['occ_cam'] = occ_cam
        # meta['occ_world'] = occ_world

        # convert_data.convert_da_pts_color(occ_world, color_world, 'label_world')
        # convert_data.convert_da_pts_color(occ_cam, color_world, 'label_cam')

        data_tuple = (imgs, meta, occs)
        self.base_time_toc_4 = time.time()

        # print(f"base 0: {(self.base_time_toc_1-self.base_time_toc_0):.6f}s, "
        #       f"base 1: {(self.base_time_toc_2-self.base_time_toc_1):.6f}s, "
        #       f"base 2: {(self.base_time_toc_3-self.vggt_time_toc):.6f}s, "
        #       f"base 2: {(self.base_time_toc_4 - self.depth_time_toc):.6f}s " )
        # print(f"vggt: {(self.vggt_time_toc-self.base_time_toc_2):6f}s, "
        #       f"depth: {(self.depth_time_toc-self.base_time_toc_3):6f}s, \n"
        #       f"total: {(self.base_time_toc_4-self.base_time_toc_0):6f}s")

        return data_tuple

    def pix2cam(self, depth, cam_k):
        h, w = depth.shape[:2]  # [480, 640]
        fx, fy, cx, cy = cam_k[0, 0], cam_k[1, 1], cam_k[0, 2], cam_k[1, 2]

        # Create coordinate grids using numpy
        v, u = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")  # [H, W]
        u = u.astype(float).flatten()  # [H*w]
        v = v.astype(float).flatten()  # [H*W]

        da_pred_flatten = depth.flatten()
        X = (u - cx) * da_pred_flatten / fx
        Y = (v - cy) * da_pred_flatten / fy
        Z = da_pred_flatten

        pts_cam = np.stack((X, Y, Z), axis=0)  # [3, H*W]
        pts_cam_t = pts_cam.T  # [H*W, 3]

        return pts_cam_t

    def cam2world(self, pts, cam2world):
        # pts shape should be [H*W, 3]
        ones = np.ones((1, pts.shape[0]))
        pts_cam_hom = np.vstack([pts.T, ones])  # [4, (H*W)']
        world_coords = cam2world @ pts_cam_hom  # [4, (H*W)']
        pts_world = world_coords[:3, :]  # [3, (H*W)']
        pts_world_t = pts_world.T  # [(H*W)', 3]

        return pts_world_t

    def depth2world(self, depth, cam_k, cam2world):
        # Ensure depth is a numpy array
        if not isinstance(depth, np.ndarray):
            depth = np.array(depth)

        # Create valid mask
        valid_mask = depth != 0  # [H, W]

        # Flatten the mask to match the point length (H*W,)
        valid_mask_flat = valid_mask.flatten()  # [H*W]

        # Convert all pixels to camera coordinates first
        points_cam = self.pix2cam(depth, cam_k)  # [H*W, 3]

        # Transform points to world coordinates
        points_world = self.cam2world(points_cam, cam2world)  # [H*W, 3]

        # Return with flattened mask that has the same length as points
        return points_world, points_cam, valid_mask_flat

    def get_meshgrid(self, ranges, grid, reso):
        pass
    
    def get_data_info(self, info):
        pass

    def get_scene_index(self, scene_name=None):
        pass
