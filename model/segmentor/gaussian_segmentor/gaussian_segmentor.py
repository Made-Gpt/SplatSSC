import time
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.cm as cm
import torch.nn.functional as F_
import matplotlib.pyplot as plt

from .utils import ConvertData, OccProbDebug

from copy import deepcopy
from mmengine import Config
from mmengine import build_from_cfg
from mmengine.model import BaseModule
from mmengine.registry import MODELS
from model.serialization import CustomSerialization
from model.depthbranch.unet2d import DecoderBN

from Depth_Anything_V2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2


@MODELS.register_module()
class GaussianSegmentor(CustomSerialization):

    def __init__(
        self,
        coding_param: dict,
        flag_depthbranch=True,  # False: use depth branch in ISO; True: use our depth branch
        zero_shot_ckpt=None,
        supervision=True,
        stage_tg='main',
        save_da=None,
        backbone=None,
        neck=None,
        lifter=None,
        encoder=None,
        head=None,
        **kwargs,
    ):
        super().__init__(**coding_param)
        self.save_da = save_da
        self.stage_tg = stage_tg
        self.supervision = supervision
        self.zero_shot_ckpt = zero_shot_ckpt
        self.cmap = cm.get_cmap('Spectral_r')
        self.max_depth = 20

        self.flag_depthbranch = flag_depthbranch
        self.efficiency_count = {}

        if self.supervision:
            cuda_kwargs = dict(
                scale_multiplier=3,
                H=60, W=60, D=36,
                pc_min=[-51.2, -51.2, -5.0],
                grid_size=0.08)
            self.occ_debug_er = OccProbDebug(
                grid_shape=(60, 60, 36),
                grid_size=0.08,
                epsilon=1e-6,
                max_clamp=15.0,
                loss_type='geo_3',  # 'sem-geo_3',
                cuda_kwargs=cuda_kwargs,
            )

        # zero-shot depth estimate model
        self.pretrained_model_type = 'da'
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        self.pretrained_model = DepthAnythingV2(**{**model_configs['vitb'], 'max_depth': self.max_depth})
        if 'finetune' in zero_shot_ckpt:
            # self.flag_depthbranch = False
            checkpoint = torch.load(zero_shot_ckpt, map_location='cpu')['model']
            new_state_dict = {}
            for k, v in checkpoint.items():
                if k.startswith('module.'):
                    new_key = k[len('module.'):]
                else:
                    new_key = k
                new_state_dict[new_key] = v
            self.pretrained_model.load_state_dict(new_state_dict)
        else:
            checkpoint = torch.load(zero_shot_ckpt, map_location='cpu')
            self.pretrained_model.load_state_dict(checkpoint)

        # image encoder
        basemodel_name = "tf_efficientnet_b7_ns"
        num_features = 2560
        print("Loading base model ()...".format(basemodel_name), end="")
        basemodel = torch.hub.load(
            "rwightman/gen-efficientnet-pytorch", basemodel_name, pretrained=True
        )
        print("Done.")
        # Remove last layer
        print("Removing last two layers (global_pool & classifier).")
        basemodel.global_pool = nn.Identity()
        basemodel.classifier = nn.Identity()

        self.backbone = basemodel

        self.neck = DecoderBN(
            out_feature=96,
            use_decoder=True,
            bottleneck_features=num_features,
            num_features=num_features,
        )

        if lifter is not None:
            self.lifter = MODELS.build(lifter)
        if encoder is not None:
            self.encoder = MODELS.build(encoder)
        if head is not None:
            self.head = MODELS.build(head)

    def extract_img_feat(self, imgs):
        # Downloading: "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b7-dcc49843.pth" to /home/wyq/.cache/torch/hub/checkpoints/efficientnet-b7-dcc49843.pth
        B, N, C, H, W = imgs.size()
        imgs = imgs.reshape(B * N, C, H, W) # 1, 3, 480, 640

        strat_backbone_toc = time.time()
        feature_x = [imgs]
        feature_idx = 0
        this_x = feature_x[-1]
        for k, v in self.backbone._modules.items():
            if k == "blocks":
                for ki, vi in v._modules.items():
                    this_x = vi(this_x)
                    feature_idx += 1
                    if feature_idx in [4, 5, 6, 8, 11]:
                        feature_x.append(this_x)
            else:
                this_x = v(this_x)
                feature_idx += 1
                if feature_idx in [4, 5, 6, 8, 11]:
                    feature_x.append(this_x)
        self.backbone_toc = time.time() - strat_backbone_toc

        img_feats_backbone = feature_x

        # list of [2560, 15, 20]
        strat_neck_toc = time.time()
        img_feats_out = self.neck(img_feats_backbone) # dict
        self.neck_toc = time.time() - strat_neck_toc

        img_feats_reshaped = []
        for img_feat in img_feats_out.values():
            BN, C, H, W = img_feat.size()
            if W != 640:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))

        return img_feats_reshaped, img_feats_out['1_1'] # list of [1, 1, 96, 28, 36], [1, 1, 96, 14, 18], [1, 1, 96, 7, 9]

    @staticmethod
    def feature_maps_format(feature_maps):
        bs, num_cams = feature_maps[0].shape[:2]
        spatial_shape = []
        scale_start_index = [0]

        col_feats = []
        for i, feat in enumerate(feature_maps):
            spatial_shape.append(feat.shape[-2:])
            scale_start_index.append(
                feat.shape[-1] * feat.shape[-2] + scale_start_index[-1]
            )
            col_feats.append(torch.reshape(
                feat, (bs, num_cams, feat.shape[2], -1)
            ))
        scale_start_index.pop()
        col_feats = torch.cat(col_feats, dim=-1).permute(0, 1, 3, 2)
        feature_maps = [
            col_feats,
            torch.tensor(
                spatial_shape,
                dtype=torch.int64,
                device=col_feats.device,
            ),
            torch.tensor(
                scale_start_index,
                dtype=torch.int64,
                device=col_feats.device,
            ),
        ]

        return feature_maps

    def scene_init(self, device_type):
        self._lut_init(device_type)

    def obtain_bev(self, imgs, metas, test_mode=False):
        B, F, N, C, H, W = imgs.shape

        start_backbone_toc = time.time()
        imgs = imgs.reshape(B * F, N, C, H, W)
        mlvl_img_feats, feature_x_4 = self.extract_img_feat(imgs)  # list of [1, 1, 96, 28, 36], [1, 1, 96, 14, 18], [1, 1, 96, 7, 9]
        self.efficiency_count['backbone_toc'] = time.time() - start_backbone_toc
        feature_maps = self.feature_maps_format(mlvl_img_feats)

        # color = imgs.squeeze(0).squeeze(0).squeeze(0).permute(1,2,0).cpu().numpy()
        # self.convert_data.convert_rgb(color, 'rgb')

        start_pretrain_toc = time.time()
        if self.pretrained_model_type == 'da':
            self.pretrained_model.eval()
            image_ = metas[0]['img_depthbranch']  # [B, 3, 490, 644]
            # depth_pred: [480, 640], depth_feat: [B, 64, 480, 640]
            depth_pred, depth_feat = self.pretrained_model.infer_image(image_, 480, 640, output_feature=True)
            depthnet_output = depth_pred  # [480, 640]
            if 'finetune' not in self.zero_shot_ckpt:
                depthnet_output = self.max_depth - depthnet_output
            depth_conf = None
        else:
            depthnet_output = None
            depth_conf = None
            depth_feat = None
        self.efficiency_count['pretrain_toc'] = time.time() - start_pretrain_toc

        ELUT = [self.EX, self.EY, self.EZ]

        start_lifter_toc = time.time()
        depth_pred, cam_pts, world_pts, sample_index, anchor, inference_feature, conf_map = self.lifter(
            metas,
            self.flag_depthbranch,
            depthnet_output,  # [B, h, w]
            feature_maps,
            depth_feat,  # [B, 1, 128, h, w]
            depth_conf,  # [B, 1, h, w]
            ELUT,
        )  # b, g, c
        self.efficiency_count['lifter_toc'] = time.time() - start_lifter_toc
        # self.convert_data.convert_da_refine_color(anchor[..., :3].squeeze(0), anchor[..., 11:22].squeeze(0), "ori_gs")

        # for training (depth branch) only
        if not test_mode and self.stage_tg=='depth_branch':
            # depth_pred: [be, h, w]
            # sample_h, sample_w = self.lifter.sample_shape
            sample_h, sample_w = depth_pred.shape[1:]

            # downsample and mask depth ground truth
            depth_gt_flatten = metas[0]['depth_gt'].flatten(1, 2)  # [bs, h * w]
            valid_depth_mask_flatten = metas[0]['valid_depth_mask'].flatten(1, 2)  # [bs, h * w]

            depth_gt_sampled = depth_gt_flatten[..., sample_index]  # [bs, sampled h * sampled w]
            valid_depth_mask_sampled = valid_depth_mask_flatten[..., sample_index]  # [bs, sampled h * sampled w]

            depth_gt_reshaped = depth_gt_sampled.reshape(B, sample_h, sample_w)  # [B, sampled h, sampled w]
            valid_depth_mask_reshaped = valid_depth_mask_sampled.reshape(B, sample_h, sample_w)  # [B, sampled h, sampled w]

            # downsample and mask point cloud ground truth
            pts_world_gt = metas[0]['pts_world_gt']  # [bs, sampled h * sampled w, 3]
            pts_world_gt_sampled = pts_world_gt[:, sample_index, :][valid_depth_mask_sampled]  # [1, n, 3]
            da_down_world_sampled = world_pts[valid_depth_mask_sampled]

            depth_branch_dict = {
                'da_depth_input': depth_pred,  # [bs, h, w]
                'da_depth_label': depth_gt_reshaped,  # [bs, h, w]
                'da_pts_input': da_down_world_sampled,  # [bs, n, 3]
                'da_pts_label': pts_world_gt_sampled,  # [bs, n, 3]
                'valid_mask': valid_depth_mask_reshaped,  # [sampled h, sampled w]
            }
        # for inference
        else:
            depth_branch_dict = {
                'da_depth_input': depth_pred,  # [1, h, w]
                'da_pts_input': world_pts.unsqueeze(0),  # [1, n, 3]
            }

        self.curr_point.update({
            'anchor': anchor,
            "feat": inference_feature,
            "conf": None,
        })

        # anchor from query point
        start_encoder_toc = time.time()
        anchor, predictions = self.encoder(
            ELUT,
            self.curr_point,  # query, aggregate and output
            feature_maps,
            metas,
        )  # b, g, c

        if hasattr(self.encoder, 'encoder_count'):
            self.efficiency_count['encoder_toc'] = time.time() - start_encoder_toc
            self.efficiency_count['encoder_sub_toc'] = self.encoder.encoder_count
        else:
            self.efficiency_count['encoder_toc'] = time.time() - start_encoder_toc

        return anchor, depth_branch_dict, predictions

    def forward(
        self,
        imgs=None,
        metas=None,
        label=None,
        test_mode=False,
        **kwargs,
    ):
        start_segmentor_toc = time.time()
        B, F, N, C, H, W = imgs.shape
        # assert B==1, 'bs > 1 not supported'

        color = deepcopy(imgs.flatten(0, 3).permute(1, 2, 0))
        self.color = color.detach().cpu().numpy()
        self.curr_name = metas[0]['name']
        self.convert_data = ConvertData(
            self.curr_name,
            self.save_da,
            self.cmap,
        )

        gaussian_anchor, depth_branch_dict, predictions = self.obtain_bev(imgs, metas, test_mode)
        # if self.supervision:
        #     print(f"::: depth branch: {self.depth_branch_toc:.5f}s overall")

        start_head_toc = time.time()
        output_dict = dict()
        output_dict = self.head(
            anchor=gaussian_anchor,  # [1, 21600, 24]
            prediction=predictions,
            label=label,
            output_dict=output_dict,
            metas=metas,
            test_mode=test_mode)

        self.efficiency_count['head_toc'] = time.time() - start_head_toc
        self.efficiency_count['segmentor_toc'] = time.time() - start_segmentor_toc

        output_dict.update(depth_branch_dict)  # add depth branch result

        if test_mode:
            if self.supervision:
                gt_xyz = metas[0]['occ_xyz'].unsqueeze(0)
                sampled_xyz = gt_xyz.flatten(1, 3).float()
                nyu_pc_min = metas[0]['vox_origin'].to(torch.float32).to(sampled_xyz.device)
                print(f"Processing {self.curr_name} ...")
                results_df = self.occ_debug_er.compute_probability_scale_loss(
                    output_dict=output_dict,
                    nyu_pc_min=nyu_pc_min,
                    sampled_xyz=sampled_xyz,
                    convert_list=[0, 1, 2],
                    radii=[0.16],
                    convert_data=self.convert_data,
                    log_view=False,
                )

        return output_dict

    @staticmethod
    def supervise_toc(current_toc, last_toc=None, denominator=1, log_view=True):
        # print & count
        overall_toc = current_toc['segmentor_toc'] / denominator
        if log_view: print(f"{24 * '-'} inference time: {overall_toc:.4f} {24 * '-'}")
        for key in current_toc.keys():
            if key == 'segmentor_toc':
                if last_toc is not None:
                    if key in last_toc.keys():
                        last_toc[key] += current_toc[key]
                    else:
                        last_toc[key] = 0
                        last_toc[key] += current_toc[key]
                continue

            if current_toc[key] is not None:
                if isinstance(current_toc[key], dict):
                    for subkey in current_toc[key].keys():
                        if last_toc is not None:
                            if key in last_toc.keys():
                                last_toc[subkey] += current_toc[key][subkey]
                            else:
                                last_toc[subkey] = 0
                                last_toc[subkey] += current_toc[key][subkey]
                        time_toc = current_toc[key][subkey] / denominator
                        if log_view: print(f"{3 * ' '} {subkey} time: {time_toc:.5f}s")
                else:
                    if last_toc is not None:
                        if key in last_toc.keys():
                            last_toc[key] += current_toc[key]
                        else:
                            last_toc[key] = 0
                            last_toc[key] += current_toc[key]
                    time_toc = current_toc[key] / denominator
                    if log_view: print(f"::: {key} time: {time_toc:.5f}s")

        if last_toc is not None:
            return last_toc

