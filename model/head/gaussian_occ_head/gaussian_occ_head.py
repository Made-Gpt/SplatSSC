import torch, torch.nn as nn
import torch.nn.functional as F
from mmengine import MODELS
from mmengine.model import BaseModule
from triton.ops.blocksparse import softmax

from ...encoder.gaussianformer.utils import \
    cartesian, safe_sigmoid, GaussianPrediction, get_rotation_matrix, safe_get_quaternion
from ...segmentor.gaussian_segmentor.utils import ConvertData

import sys
import time
import matplotlib.pyplot as plt


class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'  # Reset to default


def print_matrix_table(matrix, name, highlight_diagonal=True):
    name_str = f"{Colors.BOLD}{Colors.MAGENTA}{name}{Colors.END}"
    print(f"{name_str} (min~max):")
    print("    ", end="")
    for j in range(matrix.shape[-2]):
        print(f"{f'col {j}':<25}", end="")
    print()

    for i in range(matrix.shape[-1]):
        print(f"r{i}: ", end="")
        for j in range(matrix.shape[-2]):
            min_val = matrix[:, :, i, j].min()
            max_val = matrix[:, :, i, j].max()
            element_str = f"{min_val:.4f}~{max_val:.4f}"

            if highlight_diagonal and i == j:
                loc_str = f"{Colors.BOLD}{Colors.CYAN}{element_str}{Colors.END}"
                formatted_str = loc_str + ' ' * (25 - len(element_str))
            else:
                formatted_str = f"{element_str:<25}"
            print(formatted_str, end="")
        print()


@MODELS.register_module()
class GaussianOccHead(BaseModule):
    def __init__(
        self,
        save_da=None,
        norm_map=False,
        empty_idx=12,  # 12
        ignore_label=0,
        num_classes=11,  # 13, 12: exclude 'empty'; 11: exclude 'empty' and 'ignore'
        cuda_kwargs=dict(
            scale_multiplier=3,
            H=200, W=200, D=16,
            pc_min=[-40.0, -40.0, -1.0],
            grid_size=0.4),
        pc_range=[],
        scale_range=[],
        opas_thresh=0.09,
        include_opa=True,
        prob_loss_type='geo_3',
        aggregate_mode='local_aggregate_prob',  # 'local_aggregate_prob_new'
        coordinate_boundary='mask',  # 'clamp'
        semantics_activation='softmax'
    ):
        super().__init__()

        self.save_da = save_da
        self.norm_map = norm_map
        self.empty_idx = empty_idx
        self.ignore_label = ignore_label
        self.num_classes = num_classes  # 11
        self.classes = list(range(num_classes))  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        self.cuda_kwargs = cuda_kwargs
        # sys.path.append('/EmbodiedOcc/model/head/gaussian_occ_head/ops/localagg')
        if aggregate_mode == 'local_aggregate_prob':
            from local_aggregate_prob import LocalAggregator
            self.aggregator = LocalAggregator(**cuda_kwargs)
        elif aggregate_mode == 'local_aggregate_prob_new':
            from local_aggregate_prob_new import LocalAggregator
            self.aggregator = LocalAggregator(**cuda_kwargs)
        elif aggregate_mode == 'local_aggregate_prob_opa':
            from local_aggregate_prob_opa import LocalAggregator
            self.aggregator = LocalAggregator(**cuda_kwargs)
        else:
            from local_aggregate import LocalAggregator
            self.aggregator = LocalAggregator(**cuda_kwargs)

        self.pc_range = pc_range
        self.scale_range = scale_range
        self.opas_thresh = opas_thresh
        self.include_opa = include_opa
        self.semantic_start = 10 + int(include_opa)
        self.semantic_dim = self.num_classes

        prob_loss_str = prob_loss_type.split('_')  # eg, 'sem_3' --> ['sem', '3']
        self.sem_or_geo = prob_loss_str[0]  # 'sem' or 'geo'
        self.cache_num = int(prob_loss_str[1]) # [1, 2, 3]

        self.aggregate_mode = aggregate_mode
        self.coordinate_boundary = coordinate_boundary
        self.semantics_activation = semantics_activation

    def prepare_gaussian_args(self, anchor, metas):
        epsilon = 1e-3
        nyu_pc_min = metas[0]['vox_origin'].unsqueeze(1)  # [bs, 1, 3]
        nyu_pc_max = nyu_pc_min + metas[0]['scene_size'].unsqueeze(1)  # [bs, 1, 3]
        nyu_vox_range = metas[0]['nyu_vox_range'].to(torch.float32).unsqueeze(1)  # [bs, 1, 6]
        cam2world = metas[0]['cam2world'].to(torch.float32)  # [bs, 4, 4]

        means_cam = cartesian(anchor, nyu_vox_range, norm_map=True)  # b, g, 3
        b_, g_, _ = means_cam.shape
        # convert xyz to world coordinate
        # means_cam = means_cam.reshape(-1, 3)  # b * g, 3
        # means_cam_ = torch.cat((means_cam, torch.ones((means_cam.shape[0], 1), device=means_cam.device)), dim=1).to(torch.float32)
        # means_world_ = (cam2world @ means_cam_.unsqueeze(-1)).squeeze(-1)
        # means_world = means_world_[:, :3]
        # means_world = means_world.reshape(b_, g_, 3)
        # means = means_world

        # My Fix
        means_cam_homo = torch.cat((means_cam, torch.ones((b_, g_, 1), device=means_cam.device)), dim=-1).to(torch.float32)  # [b, num, 4]
        means_world_homo = cam2world @ means_cam_homo.transpose(1, 2)  # [b, 4, num]
        means_world = means_world_homo.transpose(1, 2)[..., :3]  # [b, num, 3]
        means = means_world
        # My Fix

        # activate scale, semantic and opacity
        rotations = anchor[..., 6: 10]  # [b, num, 4]
        gs_scales = safe_sigmoid(anchor[..., 3:6])  # [b, num, 3]
        scales = self.scale_range[0] + (self.scale_range[1] - self.scale_range[0]) * gs_scales

        semantics = anchor[..., self.semantic_start: (self.semantic_start + self.semantic_dim)]  # [b, num c]
        shs = torch.zeros(*anchor.shape[:-1], 0, device=anchor.device, dtype=anchor.dtype)
        opacities = safe_sigmoid(anchor[..., 10: (10 + int(self.include_opa))])  # [b, num, 1]
        if opacities.numel() == 0:
            opacities = torch.ones_like(semantics[..., :1], requires_grad=False)

        # mask in world coordinate
        if self.coordinate_boundary == 'mask':
            # mask_gaussian: [bs, num]
            mask_gaussian = ((means[..., 0] > (nyu_pc_min[..., 0] + epsilon)) & (means[..., 0] < (nyu_pc_max[..., 0] - epsilon)) &
                             (means[..., 1] > (nyu_pc_min[..., 1] + epsilon)) & (means[..., 1] < (nyu_pc_max[..., 1] - epsilon)) &
                             (means[..., 2] > (nyu_pc_min[..., 2] + epsilon)) & (means[..., 2] < (nyu_pc_max[..., 2] - epsilon)))
            # means = means[mask_gaussian].unsqueeze(0)  # [1, num, 3]
            # scales = scales[mask_gaussian].unsqueeze(0)  # [1, num, 3]
            # rotations = rotations[mask_gaussian].unsqueeze(0)  # [1, num, 4]
            # opacities = opacities[mask_gaussian].unsqueeze(0)  # [1, num, 1]
            # semantics = semantics[mask_gaussian].unsqueeze(0)  # [1, num, c]
            mask_gaussian = mask_gaussian.unsqueeze(-1)  # [bs, num, 1]
            means = means * mask_gaussian  # [1, num, 3]
            scales = scales * mask_gaussian  # [1, num, 3]
            rotations = rotations * mask_gaussian  # [1, num, 4]
            opacities = opacities * mask_gaussian  # [1, num, 1]
            semantics = semantics * mask_gaussian  # [1, num, c]
        elif self.coordinate_boundary == 'clamp':
            means = torch.clamp(means, nyu_pc_min + epsilon, nyu_pc_max - epsilon)

        # logit or probability
        if self.semantics_activation == 'softmax':
            semantics = semantics.softmax(dim=-1)
        elif self.semantics_activation == 'softplus':
            semantics = F.softplus(semantics)
        elif self.semantics_activation == 'sigmoid':
            semantics = safe_sigmoid(semantics)

        # in world coordinate (for test and train)
        gaussian = GaussianPrediction(
            means=means,  # [bs, num, 3]
            scales=scales,  # [bs, num, 3]
            rotations=rotations,  # [bs, num, 4]
            harmonics=shs.unflatten(-1, (3, -1)),
            opacities=opacities,  # [bs, num, 1]
            semantics=semantics,  # [bs, num, c]
        )

        # in cam coordinate (for visualization)
        gaussian_cam = GaussianPrediction(
            means=means_cam,  # [bs, num, 3]
            scales=scales,  # [bs, num, 3]
            rotations=rotations,  # [bs, num, 4]
            harmonics=shs.unflatten(-1, (3, -1)),
            opacities=opacities,  # [bs, num, 1]
            semantics=semantics,  # [bs, num, c]
        )

        b_m, g_m, _ = means.shape
        # diagonal matrix
        S = torch.zeros(b_m, g_m, 3, 3, dtype=means.dtype, device=means.device)
        S[..., 0, 0] = scales[..., 0]
        S[..., 1, 1] = scales[..., 1]
        S[..., 2, 2] = scales[..., 2]
        # print_matrix_table(S, 'scale matrix')

        R = get_rotation_matrix(rotations)  # b, g, 3, 3
        # print_matrix_table(R, 'rotation matrix')
        M = torch.matmul(R, S)  # M = RS
        Cov = torch.matmul(M, M.transpose(-1, -2))  # Cov = M M^T = RS (RS)^T = RS S^T R^T = R S^2 R^T
        # print_matrix_table(Cov, 'covariance matrix')

        # c2w_rot = cam2world[:3, :3]  # [bs, 3, 3]
        # c2w_rot_T = cam2world[:3, :3].T
        # c2w_rot = c2w_rot.unsqueeze(0).unsqueeze(0).repeat(b_m, g_m, 1, 1).to(torch.float32)
        # c2w_rot_T = c2w_rot_T.unsqueeze(0).unsqueeze(0).repeat(b_m, g_m, 1, 1).to(torch.float32)
        # Cov = torch.matmul(c2w_rot, torch.matmul(Cov, c2w_rot_T))
        # print_matrix_table(Cov, 'covariance world matrix')

        c2w_rot = cam2world[:, :3, :3]  # [bs, 3, 3]
        c2w_rot_ = c2w_rot.unsqueeze(1)  # [bs, 1, 3, 3]
        cov_trans = c2w_rot_ @ Cov @ c2w_rot_.transpose(-2, -1)
        cov_inv = cov_trans.to(torch.float32).inverse()  # b, g, 3, 3

        # cov_inv = Cov.to(torch.float32).inverse()  # b, g, 3, 3
        # print_matrix_table(cov_inv, 'covariance invariance matrix')

        return gaussian, gaussian_cam, cov_inv

    def forward(self, anchor, prediction:list, label, output_dict, metas, test_mode=False):

        # assert anchor.shape[0] == 1, 'only support for single camera prediction'
        self.curr_name = metas[0]['name']
        self.convert_data = ConvertData(
            self.curr_name,
            self.save_da,
        )

        gt_xyz = metas[0]['occ_xyz']  # [bs, x, y, z, 3] --> [bs, 60, 60, 36, 3]
        origin_use = metas[0]['vox_origin'].to(torch.float32).to(anchor.device)  # [bs, 3]
        fov_mask = metas[0]['fov_mask']  # [bs, 60, 60, 36]
        sampled_xyz = gt_xyz.flatten(1, 3).float()  # [bs, 129600, 3]
        # label: [bs, 1, 60, 60, 36]
        spatial_shape = label.shape[2:]  # [60, 60, 36]
        ssc_target = label.squeeze(1)  # [bs, 60, 60, 36]

        # prepare gaussian args in world coordinate
        B, G, _ = anchor.shape
        gaussian_cam_cache = []
        gaussian_cache = []
        cov_inv_cache = []
        sem_cache = []

        # for i in range(len(prediction)):
        for i in range(self.cache_num):
            gaussian, gaussian_cam, cov_inv = self.prepare_gaussian_args(prediction[i]['anchor'], metas)
            gaussian_cam_cache.append(gaussian_cam)
            gaussian_cache.append(gaussian)
            cov_inv_cache.append(cov_inv)

            # for geometry supervision, no need to store all semantic cache, only leave the latest one
            if self.sem_or_geo != 'sem' and i < (self.cache_num - 1):
                continue

            # gaussian_cam = gaussian_cam_cache[-1]
            # gaussian = gaussian_cache[-1]
            cov_inv = cov_inv_cache[-1]  # [bs, num, 3, 3]
            means = gaussian.means  # [bs, num, 3]
            origi_opa = gaussian.semantics  # [bs, num, dim]
            opacities = gaussian.opacities.flatten(1, 2)  # [bs, num]
            scales = gaussian.scales  # [bs, num, 3]

            # aggregation
            semantics = []
            opas_thresh = torch.tensor(self.opas_thresh, device=means.device)
            for i in range(len(sampled_xyz)):  # batch size
                semantic = self.aggregator(
                    sampled_xyz[i:(i+1)],
                    means[i:(i+1)],
                    opacities[i:(i+1)],
                    origi_opa[i:(i+1)],
                    scales[i:(i+1)],
                    cov_inv[i:(i+1)],
                    origin_use[i:(i+1)],
                    opas_thresh,
                )  # n, c

                if self.aggregate_mode == 'local_aggregate':
                    semantics.append(semantic)
                else:
                    sem = semantic[0] * semantic[1].unsqueeze(-1)  # n, c
                    geo_logit = 1 - semantic[1].unsqueeze(-1)  # n, 1
                    geo_sem = torch.cat([geo_logit, sem], dim=-1)  # cat geo prob as 'empty' -- '0'
                    semantics.append(geo_sem)
                    # self.occ_prob_o3d(geo_logit, origin_use)
            semantics = torch.stack(semantics, dim=0).transpose(1, 2)  # [bs, 12, 129600]
            semantics = semantics.unflatten(-1, spatial_shape)  # [bs, 12, 60, 60, 36]
            sem_cache.append(semantics)

        gaussian_cam = gaussian_cam_cache[-1]
        gaussian = gaussian_cache[-1]
        semantics = sem_cache[-1]
        result_dict = {
            'ce_input': semantics,                               # [bs, 12, 60, 60, 36]
            'ce_label': ssc_target,                              # [bs, 60, 60, 36]
            'fov_mask': fov_mask,                                # [bs, 60, 60, 36]
            'pc_min': origin_use,                                # [bs, 3]
            'sampled_xyz': sampled_xyz,                          # [bs, 129600, 3]

            'gaussian': gaussian,                                # in world coordinate
            'gaussian_cache': gaussian_cache,                    # list, world coordinate
            'cov_inv_cache': cov_inv_cache,
            'sem_cache': sem_cache,
        }
        output_dict.update(result_dict)

        # visualization
        # self.convert_data.convert_gaussian_ellipsoids(gaussian_cam, gaussian_cam.semantics.squeeze(0), alpha=1, name="gaussian_cam")
        # self.convert_data.convert_gaussian_ellipsoids(gaussian_cam_cache[1], gaussian_cam.semantics.squeeze(0), alpha=1, name="gaussian_cam_1")
        # self.convert_data.convert_gaussian_ellipsoids(gaussian_cam_cache[0], gaussian_cam.semantics.squeeze(0), alpha=1, name="gaussian_cam_0")
        # self.convert_data.convert_gaussian_ellipsoids(gaussian, origi_opa.squeeze(0), name="gaussian")
        # self.convert_data.convert_da_refine_color(means.squeeze(0), origi_opa.squeeze(0), name="mean_sem")
        # self.convert_data.convert_occ_label_color(result_dict['ce_input'], 0.08, metas[0]['vox_origin'], name='pred')
        # self.convert_data.convert_occ_label_color(result_dict['ce_input'], 0.08, metas[0]['vox_origin'], name='pred_agg1')
        # self.convert_data.convert_occ_label_color(result_dict['ce_label'], 0.08, metas[0]['vox_origin'], name='label')

        # world2cam = metas[0]['world2cam'].to(torch.float32)
        # self.convert_data.convert_occ_label_color(result_dict['ce_input'], 0.08, metas[0]['vox_origin'], name='pred_cam', matrix=world2cam.squeeze(0))
        # self.convert_data.convert_occ_label_color(result_dict['ce_label'], 0.08, metas[0]['vox_origin'], name='label_cam', matrix=world2cam.squeeze(0))

        # cam_k = metas[0]['cam_k'].to(torch.float32)
        # cam2world = metas[0]['cam2world'].to(torch.float32)
        # self.convert_data.convert_camera_info(cam2world, cam_k, name='camera')

        return output_dict

    def occ_prob_o3d(self, geo_logit, origin_use):
        x_coords = torch.arange(60, device=geo_logit.device) * 0.08 + origin_use[0]
        y_coords = torch.arange(60, device=geo_logit.device) * 0.08 + origin_use[1]
        z_coords = torch.arange(36, device=geo_logit.device) * 0.08 + origin_use[2]
        X, Y, Z = torch.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        points = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1)

        mask = (geo_logit < 0.199).squeeze()
        points = points[mask]

        cmap = plt.cm.coolwarm
        colors = cmap((1 - geo_logit).cpu().numpy().squeeze())[:, :3]  # [129600, 3] RGB values
        colors = colors[mask.cpu().numpy()]
        self.convert_data.convert_da_pts_color(points, colors, 'bin')

    def occ_prob_loss(self, pred, ssc_target, fov_mask):
        pred_probs = pred.float()
        ssc_target = ssc_target.long()  # ground truth, [1, 60, 60, 36]
        ssc_bin = ((ssc_target != self.ignore_label) & (ssc_target != self.empty_idx)).to(torch.uint8)  # [12, 0] --> 0, [else] --> 1.

        mask = ssc_target != self.ignore_label  # 'ignore' area: 0, 'empty' + 'valid' area: 1
        if fov_mask is not None:
            mask = mask & fov_mask

        # mask gt, [0, 1]
        valid_target = ssc_bin[mask].float()  # get non-ignore area, where 'empty' area: 0, 'valid' area: 1
        valid_probs = pred_probs[mask]  # get non-ignore area

        loss = F.binary_cross_entropy(valid_probs, valid_target)

        return loss

