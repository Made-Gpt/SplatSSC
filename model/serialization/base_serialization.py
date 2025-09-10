import time
import torch
from mmengine import MODELS
from model.serialization.point_transformer.serialization.custom import CustomCoding


@torch.inference_mode()
def offset2bincount(offset):
    """ transport offset to point number in each batch """
    return torch.diff(
        offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long)
    )  # calculate diff between 2 adjacent offset: diff[i]=offset[i]-offset[i-1]


@torch.inference_mode()
def offset2batch(offset):
    bincount = offset2bincount(offset)
    return torch.arange(
        len(bincount), device=offset.device, dtype=torch.long
    ).repeat_interleave(bincount)


@torch.inference_mode()
def batch2offset(batch):
    return torch.cumsum(batch.bincount(), dim=0).long()


key_configs = {'r32': 31, 'r64': 63, 'r128': 127, 'r256': 255, 'r512': 511,}


@MODELS.register_module()
class CustomSerialization(CustomCoding):
    def __init__(
            self,
            depth=16,
            order='z',
            lut_size=None,
            shuffle_orders=False,
            log_view=True,
    ):
        super(CustomSerialization, self).__init__(lut_size=lut_size)
        self.depth = depth
        self.order = order
        self.shuffle_orders = shuffle_orders

        self.curr_point = {}
        self.hist_point = {}
        self.reused_point = {}

        self.sort_time = 0.0
        self.code_time = 0.0
        self.log_view = log_view

    @torch.no_grad()
    def _lut_init(self,
                  device_type,
                  ELUT:list=None,
                  # DLUT:list=None,
                  scene_origin=None,
                  scene_size=None,
                  ):
        # if ELUT is not None and DLUT is not None:
        if ELUT is not None:
            self.EX, self.EY, self.EZ = ELUT
            # self.DX, self.DY, self.DZ = DLUT
            self.init = True
        else:
            start_lut_init = time.time()
            self._call_init(device_type)
            # self.hist_point['anchor'] = torch.tensor([], device=device_type)
            # self.hist_point['feat'] = torch.tensor([], device=device_type)
            lut_init_time = time.time() - start_lut_init
            if self.log_view:
                print(f"\n{64*'-'}\nLUT range: {self.lut_size}, LUT Init Time: {lut_init_time:.8f}s\n{64*'-'}\n")
        self.scene_size = scene_size
        self.scene_origin = scene_origin

        # self.curr_point['anchor'] = torch.tensor([], device=device_type)
        # self.curr_point['feat'] = torch.tensor([], device=device_type)
        # self.hist_point['anchor'] = torch.tensor([], device=device_type)
        # self.hist_point['feat'] = torch.tensor([], device=device_type)
        # self.reused_point['anchor'] = torch.tensor([], device=device_type)
        # self.reused_point['feat'] = torch.tensor([], device=device_type)

        self.curr_point['anchor'] = torch.empty((0, 0, 0), device=device_type)
        self.curr_point['feat'] = torch.empty((0, 0, 0), device=device_type)
        self.curr_point['mask'] = torch.empty((0, 0, 0), device=device_type)

        self.hist_point['anchor'] = torch.empty((0, 0, 0), device=device_type)
        self.hist_point['feat'] = torch.empty((0, 0, 0), device=device_type)
        self.hist_point['mask'] = torch.empty((0, 0, 0), device=device_type)

        self.reused_point['anchor'] = torch.empty((0, 0, 0), device=device_type)
        self.reused_point['feat'] = torch.empty((0, 0, 0), device=device_type)
        self.reused_point['mask'] = torch.empty((0, 0, 0), device=device_type)

    def _lut_reset(self, device_type):
        self.reused_point['anchor'] = torch.empty((0, 0, 0), device=device_type)
        self.reused_point['feat'] = torch.empty((0, 0, 0), device=device_type)
        self.reused_point['mask'] = torch.empty((0, 0, 0), device=device_type)

        self.curr_point['anchor'] = torch.empty((0, 0, 0), device=device_type)
        self.curr_point['feat'] = torch.empty((0, 0, 0), device=device_type)
        self.curr_point['mask'] = torch.empty((0, 0, 0), device=device_type)

    @staticmethod
    def voxel_cam_filter(pts_cam, cam_vox_range):
        # pts_cam [num_anchor, 3], in camera coordinate
        gaussian_cam_mask = ((pts_cam[..., 0] >= cam_vox_range[0]) & (pts_cam[..., 0] <= cam_vox_range[3]) &
                             (pts_cam[..., 1] >= cam_vox_range[1]) & (pts_cam[..., 1] <= cam_vox_range[4]) &
                             (pts_cam[..., 2] >= cam_vox_range[2]) & (pts_cam[..., 2] <= cam_vox_range[5]))
        return gaussian_cam_mask

    @staticmethod
    def voxel_world_filter(pts_world, vox_near_world, vox_far_world, epsilon):
        # pts_world [num_anchor, 3], in world coordinate
        world_mask = ((pts_world[..., 0] > (vox_near_world[0] + epsilon)) & (pts_world[..., 0] < (vox_far_world[0] - epsilon)) &
                      (pts_world[..., 1] > (vox_near_world[1] + epsilon)) & (pts_world[..., 1] < (vox_far_world[1] - epsilon)) &
                      (pts_world[..., 2] > (vox_near_world[2] + epsilon)) & (pts_world[..., 2] < (vox_far_world[2] - epsilon)))
        return world_mask

    def cam_to_world(self, pts_cam, cam2world, vox_range, epsilon, limit='filter'):
        if pts_cam.dim() == 3:
            B, N, _ = pts_cam.shape
            ones = torch.ones((B, N, 1), device=pts_cam.device)
            pts_cam_hom = torch.cat([pts_cam, ones], dim=-1).transpose(1, 2)  # [B, 4, N]
            pts_world_hom = torch.bmm(cam2world, pts_cam_hom)  # [B, 4, N]
            pts_world = pts_world_hom[:, :3, :].transpose(1, 2)  # [B, N, 3]
            if limit == 'filter':
                mask = self.voxel_world_filter(pts_world, vox_range[:3], vox_range[3:], epsilon)
                pts_world = pts_world[mask]
            elif limit == 'clamp':
                pts_world = torch.clamp(pts_world, vox_range[:3], vox_range[3:])
            return pts_world

        elif pts_cam.dim() == 2:
            ones = torch.ones((pts_cam.shape[0], 1), device=pts_cam.device)
            pts_cam_hom = torch.cat([pts_cam, ones], dim=-1).T  # [4, N]
            pts_world_hom = cam2world @ pts_cam_hom  # [4, N]
            pts_world = pts_world_hom[:3, :].T  # [N, 3]
            if limit == 'filter':
                mask = self.voxel_world_filter(pts_world, vox_range[:3], vox_range[3:], epsilon)
                pts_world = pts_world[mask]
            elif limit == 'clamp':
                pts_world = torch.clamp(pts_world, vox_range[:3], vox_range[3:])
            return pts_world
        else:
            raise ValueError(f"Unsupported pts_cam shape: {pts_cam.shape}")

    def pix_mask(self, pts_cam, cam_k, img_wh=(640, 480)):
        if cam_k.dim() == 2:
            # batch_size = 1
            fx = cam_k[0, 0]
            fy = cam_k[1, 1]
            cx = cam_k[0, 2]
            cy = cam_k[1, 2]
            x, y, z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
            u = fx * x / z + cx
            v = fy * y / z + cy

            w, h = img_wh
            mask = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            return pts_cam[mask], mask

        elif cam_k.dim() == 3:
            # batch > 1
            B, N, _ = pts_cam.shape
            fx = cam_k[:, 0, 0].view(B, 1)
            fy = cam_k[:, 1, 1].view(B, 1)
            cx = cam_k[:, 0, 2].view(B, 1)
            cy = cam_k[:, 1, 2].view(B, 1)

            x, y, z = pts_cam[..., 0], pts_cam[..., 1], pts_cam[..., 2]
            u = fx * x / z + cx
            v = fy * y / z + cy

            w, h = img_wh
            mask = (u >= 0) & (u < w) & (v >= 0) & (v < h)

            masked_pts_list = [pts_cam[b][mask[b]] for b in range(B)]
            return masked_pts_list, mask

        else:
            raise ValueError(f"Unsupported cam_k shape: {cam_k.shape}")

    def pix2cam(self, depth, cam_k):
        if depth.dim() == 2:
            # batch size == 1
            H, W = depth.shape
            fx, fy, cx, cy = cam_k[0, 0], cam_k[1, 1], cam_k[0, 2], cam_k[1, 2]
            v, u = torch.meshgrid(torch.arange(H, device=depth.device),
                                  torch.arange(W, device=depth.device),
                                  indexing="ij")
            u = u.flatten().float()
            v = v.flatten().float()
            z = depth.flatten()
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            return torch.stack([x, y, z], dim=1)  # [H*W, 3]

        elif depth.dim() == 3:
            # batch size > 1
            B, H, W = depth.shape
            v, u = torch.meshgrid(torch.arange(H, device=depth.device),
                                  torch.arange(W, device=depth.device),
                                  indexing="ij")
            u = u.flatten().float().unsqueeze(0).expand(B, -1)  # [B, H*W]
            v = v.flatten().float().unsqueeze(0).expand(B, -1)

            z = depth.view(B, -1)  # [B, H*W]
            fx = cam_k[:, 0, 0].view(B, 1)
            fy = cam_k[:, 1, 1].view(B, 1)
            cx = cam_k[:, 0, 2].view(B, 1)
            cy = cam_k[:, 1, 2].view(B, 1)

            x = (u - cx) * z / fx
            y = (v - cy) * z / fy

            return torch.stack([x, y, z], dim=2)  # [B, H*W, 3]

        else:
            raise ValueError(f"Unsupported depth shape: {depth.shape}")

    def cam2world(self, pts, cam2world):
        if pts.dim() == 2:  # batch == 1
            # pts: [N, 3], cam2world: [4, 4]
            ones = torch.ones((pts.shape[0], 1), device=pts.device)
            pts_hom = torch.cat([pts, ones], dim=-1).T  # [4, N]
            world_coords = cam2world @ pts_hom  # [4, N]
            return world_coords[:3, :].T  # [N, 3]

        elif pts.dim() == 3:
            # pts: [B, N, 3], cam2world: [B, 4, 4]
            B, N, _ = pts.shape
            ones = torch.ones((B, N, 1), device=pts.device)
            pts_hom = torch.cat([pts, ones], dim=-1)  # [B, N, 4]
            # pts_hom = pts_hom.transpose(1, 2)  # [B, 4, N]
            # world_coords = torch.bmm(cam2world, pts_hom)  # [B, 4, N]
            world_coords = cam2world @ pts_hom.transpose(1, 2)  # [B, 4, N]
            world_pts = world_coords[:, :3, :].transpose(1, 2)  # [B, N, 3]
            return world_pts

        else:
            raise ValueError(f"Unsupported pts shape: {pts.shape}. Expected [N,3] or [B,N,3].")

    @torch.no_grad()
    def serialization_pos_only(self, xyz, sort_code=True):
        """ build z-order for new anchors """
        start_code = time.time()
        # shape of data_dict["xyz"] is [1, anchor_num, 3], target shape is [anchor_num, 3]
        code = [self.custom_z_order_encode(xyz.squeeze(0), self.depth)]
        code = torch.stack(code)  # [1, anchor_num]
        self.code_time = time.time() - start_code

        if sort_code:
            start_sort = time.time()
            order = torch.argsort(code, dim=1)  # sorted index
            self.sort_time = time.time() - start_sort
            sort_code = code[:, order[0]]
            return sort_code, order
        else:
            return code, None

    @torch.no_grad()
    def serialization_pos(self, xyz):
        temp_point = {}
        """ build z-order for current anchors """
        start_code = time.time()
        # shape of data_dict["xyz"] is [1, anchor_num, 3], target shape is [anchor_num, 3]
        code = [self.custom_z_order_encode(xyz.squeeze(0), self.depth)]
        code = torch.stack(code)  # [1, anchor_num]
        code_len = code.shape[1]  # anchor_num

        start_sort = time.time()
        order = torch.argsort(code, dim=1)  # sorted index
        self.sort_time = time.time() - start_sort

        # update gaussian parameters, inverse_indices shape is [1, unique anchor num]
        unique_code = torch.unique(code, sorted=True)  # sorted, [unique anchor num]
        sorted_code = torch.gather(code, dim=1, index=order)  # [1, anchor num]
        pos = torch.searchsorted(sorted_code, unique_code.unsqueeze(0))  # [1, unique anchor num]
        pos = torch.clamp(pos, 0, code_len-1)  # Ensure pos is within valid range

        # update z-order parameters
        code = unique_code.unsqueeze(0)  # [1, unique anchor num]
        temp_point['code'] = code.requires_grad_(False)  # sorted (value), [1, anchor_num]
        self.code_time = time.time() - start_code

        return temp_point, order, pos

    @torch.no_grad()
    def serialization(self, data_dict):
        temp_point = {}
        """ build z-order for current anchors """
        start_code = time.time()
        # shape of data_dict["xyz"] is [1, anchor_num, 3], target shape is [anchor_num, 3]
        code = [self.custom_z_order_encode(data_dict['xyz'].squeeze(0), self.depth)]
        code = torch.stack(code)  # [1, anchor_num]
        code_len = code.shape[1]  # anchor_num

        start_sort = time.time()
        order = torch.argsort(code, dim=1)  # sorted index
        self.sort_time = time.time() - start_sort

        # update gaussian parameters, inverse_indices shape is [1, unique anchor num]
        unique_code = torch.unique(code, sorted=True)  # sorted, [unique anchor num]
        sorted_code = torch.gather(code, dim=1, index=order)  # [1, anchor num]
        pos = torch.searchsorted(sorted_code, unique_code.unsqueeze(0))  # [1, unique anchor num]
        pos = torch.clamp(pos, 0, code_len - 1)  # Ensure pos is within valid range

        # sort and mask
        if data_dict['conf'] is not None:
            temp_point['conf'] = data_dict['conf'][:, order[0]][:, pos[0]]  # [1, unique_anchor_num, 1] -> [1, masked_anchor_num, 1]
        else:
            temp_point['conf'] = None
        temp_point['anchor'] = data_dict['anchor'][:, order[0]][:, pos[0]]  # [1, unique_anchor_num, 23] -> [1, masked_anchor_num, 23]
        temp_point['feat'] = data_dict['feat'][:, order[0]][:, pos[0]]  # [1, unique_anchor_num, 96] -> [1, masked_anchor_num, 96]

        # update z-order parameters
        code = unique_code.unsqueeze(0)  # [1, unique anchor num]
        temp_point['code'] = code.requires_grad_(False)  # sorted (value), [1, anchor_num]
        self.code_time = time.time() - start_code

        return temp_point

    @torch.no_grad()
    def safe_norm(self, xyz, vox_range):
        xyz = (xyz - vox_range[:3]) / (vox_range[3:] - vox_range[:3])
        return xyz

    @staticmethod
    def scale_vox_range(vox_range, scale_factor):
        # shape of 'vox_range' should be [6]
        center = (vox_range[:3] + vox_range[3:]) / 2  # Compute center [1, 3]
        scaled_vox_range = scale_factor * vox_range - (scale_factor - 1) * torch.cat([center, center])
        return scaled_vox_range

    @torch.inference_mode()
    def map_to_custom(self, anchor_xyz, vox_range, lut_size):
        """ map anchor x,y,z (0,1) to grid_coord """
        xyz = self.safe_norm(anchor_xyz, vox_range)  # [0, 1]
        xxx = xyz[..., 0] * key_configs[lut_size[0]]
        yyy = xyz[..., 1] * key_configs[lut_size[1]]
        zzz = xyz[..., 2] * key_configs[lut_size[2]]
        xyz = torch.stack([xxx, yyy, zzz], dim=-1)  # [1, 21600, 3]
        return xyz

    @torch.no_grad()
    def pos_encode(self, xyz, vox_range=None, log_view=True):
        if vox_range is None:
            vox_range = torch.cat([xyz.min(dim=0).values, xyz.max(dim=0).values])
            # center = (vox_range[:3] + vox_range[3:]) / 2
            # vox_range = 2 * vox_range - torch.cat([center, center])
            vox_range = self.scale_vox_range(vox_range, 4)
        xyz = self.map_to_custom(xyz, vox_range, self.lut_size)
        temp_point, order, pos = self.serialization_pos(xyz)
        # self.curr_point.update(temp_point)  # update current point
        if log_view and self.log_view:
            print(f"\n{64*'='}\n(xyz) serialization time: {self.code_time:.5f}s, sort time: {self.sort_time:.5f}s")
        return order, pos, temp_point

    @torch.no_grad()
    def pos_encode_only(self, xyz, vox_range, sort_code=True, log_view=True):
        if vox_range is None:
            vox_range = torch.cat([xyz.min(dim=0).values, xyz.max(dim=0).values])
            vox_range = self.scale_vox_range(vox_range, 4)
        xyz = self.map_to_custom(xyz, vox_range, self.lut_size)
        code, order = self.serialization_pos_only(xyz, sort_code)
        if log_view and self.log_view:
            print(f"\n{64*'='}\n(code only) serialization time: {self.code_time:.5f}s")
        return code, order

    @torch.no_grad()
    def local_encode(self, anchor, feat, conf, vox_range=None, log_view=True):
        if vox_range is None:
            vox_range = torch.cat([anchor.min(dim=0).values, anchor.max(dim=0).values])
            vox_range = self.scale_vox_range(vox_range, 4)
        xyz = self.map_to_custom(anchor[..., :3], vox_range, self.lut_size)
        data_dict = {
            'xyz': xyz, 'feat': feat, 'anchor': anchor, 'conf': conf, 'depth': self.depth,
        }
        temp_point = self.serialization(data_dict)
        self.curr_point.update(temp_point)
        if log_view and self.log_view:
            print(f"(local) serialization time: {self.code_time:.5f}s, sort time: {self.sort_time:.5f}s")

    @torch.no_grad()
    def global_encode(self, anchor, feat, conf, vox_range, log_view=True):
        if vox_range is None:
            vox_range = torch.cat([anchor.min(dim=0).values, anchor.max(dim=0).values])
            vox_range = self.scale_vox_range(vox_range, 4)
        xyz = self.map_to_custom(anchor[..., :3], vox_range, self.lut_size)
        data_dict = {
            'xyz': xyz, 'feat': feat, 'anchor': anchor, 'conf': conf, 'depth': self.depth,
        }
        temp_point = self.serialization(data_dict)
        self.hist_point.update(temp_point)
        if log_view and self.log_view:
            print(f"(global) serialization time: {self.code_time:.5f}s, sort time: {self.sort_time:.5f}s")

    def log_xyz(self, xyz, module_num=''):
        if xyz.any():
           print(f"{18*'-'} {xyz.shape} {18*'-'}\n"
                 f"[{module_num}] x range: {xyz[:, 0].min():.5f}~{xyz[:, 0].max():.5f}, \n"
                 f"[{module_num}] y range: {xyz[:, 1].min():.5f}~{xyz[:, 1].max():.5f}, \n"
                 f"[{module_num}] z range: {xyz[:, 2].min():.5f}~{xyz[:, 2].max():.5f}; ")

