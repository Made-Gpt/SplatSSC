import time
import torch
import numpy as np
import torch.nn as nn
import matplotlib.cm as cm
import torch.nn.functional as F_
from model.segmentor.gaussian_segmentor.utils import ConvertData

from mmengine import MODELS
from mmengine.model import BaseModule
from model.serialization import CustomSerialization
from Depth_Anything_V2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2
from .unet2d import DecoderBN


@MODELS.register_module()
class FineTuneDepthBranch(CustomSerialization):
    def __init__(
        self,
        coding_param: dict,
        flag_depthanything_as_gt=False,
        zero_shot_ckpt=None,
        supervision=True,
        save_da=None,
        backbone=None,
        neck=None,
        lifter=None,
    ):
        super().__init__(**coding_param)
        self.save_da = save_da
        self.supervision = supervision
        self.flag_depthanything_as_gt = flag_depthanything_as_gt
        self.zero_shot_ckpt = zero_shot_ckpt
        self.depthbranch_count = {}
        self.cmap = cm.get_cmap('Spectral_r')

        # zero-shot depth estimate model
        self.pretrained_model_type = 'da'
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        if 'finetune' in zero_shot_ckpt:
            self.max_depth = 20
        else:
            self.max_depth = 20
        self.pretrained_model = DepthAnythingV2(**{**model_configs['vitb'], 'max_depth': self.max_depth})
        # checkpoint = torch.load(zero_shot_ckpt, map_location='cpu')
        # self.pretrained_model.load_state_dict(checkpoint)
        if 'finetune' in zero_shot_ckpt:
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
        # self.pretrained_model.eval()

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

    def extract_img_feat(self, imgs):
        # Downloading: "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b7-dcc49843.pth" to /home/wyq/.cache/torch/hub/checkpoints/efficientnet-b7-dcc49843.pth
        B, N, C, H, W = imgs.size()
        imgs = imgs.reshape(B * N, C, H, W) # 1, 3, 480, 640

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

        img_feats_backbone = feature_x

        # list of [2560, 15, 20]
        img_feats_out = self.neck(img_feats_backbone) # dict

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

    def forward(self, imgs, metas):
        start_depthbranch_toc = time.time()

        curr_name = metas[0]['name']
        convert_data = ConvertData(curr_name, self.save_da)

        B, F, N, C, H, W = imgs.shape
        imgs = imgs.reshape(B * F, N, C, H, W)

        # color = imgs.squeeze(0).squeeze(0).squeeze(0).permute(1,2,0).cpu().numpy()
        # convert_data.convert_rgb(color, 'rgb')

        start_backbone_toc = time.time()
        mlvl_img_feats, feature_x_4 = self.extract_img_feat(imgs)  # list of [1, 1, 96, 28, 36], [1, 1, 96, 14, 18], [1, 1, 96, 7, 9]
        self.depthbranch_count['backbone_toc'] = time.time() - start_backbone_toc

        start_pretrain_toc = time.time()
        if self.pretrained_model_type == 'da':
            self.pretrained_model.eval()
            image_ = metas[0]['img_depthbranch']  # [1, 3, 490, 644]
            # depth_pred: [480, 640], depth_feat: [1, 64, 480, 640]
            depth_pred, depth_feat = self.pretrained_model.infer_image(image_, 480, 640, output_feature=True)
            depthnet_output = depth_pred
            if 'finetune' not in self.zero_shot_ckpt:
                depthnet_output = self.max_depth - depthnet_output
            depth_conf = None
        else:
            depthnet_output = None
            depth_conf = None
            depth_feat = None
        self.depthbranch_count['pretrain_toc'] = time.time() - start_pretrain_toc

        ELUT = [self.EX, self.EY, self.EZ]

        # feature_maps[0] is mlvl_img_feats,
        # feature_maps[1] is spatial shape [[240, 320], [120, 160], [ 60,  80], [ 30,  40]]
        # feature_maps[2] is start index [     0,  76800,  96000, 100800]
        feature_maps = self.feature_maps_format(mlvl_img_feats)

        start_lifter_toc = time.time()
        pred, da_down_cam, da_down_world, flatten_index, inference_feature, conf_map = self.lifter(  # [1, h, w]
            metas,
            self.flag_depthanything_as_gt,
            depthnet_output,  # [1, h, w]
            feature_maps,
            depth_feat,  # [1, 1, 128, h, w]
            depth_conf,  # [1, 1, h, w]
            ELUT,
        )
        if hasattr(self.lifter, 'fine_tune_lifter_count'):
            self.depthbranch_count['lifter_toc'] = time.time() - start_lifter_toc
            self.depthbranch_count['lifter_sub_toc'] = self.lifter.fine_tune_lifter_count
        else:
            self.depthbranch_count['lifter_toc'] = time.time() - start_lifter_toc

        sample_h, sample_w = pred.shape[1:]

        # downsample and mask depth ground truth
        depth_gt_flatten = metas[0]['depth_gt'].flatten(1, 2)  # [bs, h * w]
        valid_depth_mask_flatten = metas[0]['valid_depth_mask'].flatten(1, 2)  # [bs, h * w]

        depth_gt_sampled = depth_gt_flatten[..., flatten_index]  # [bs, sampled h * sampled w]
        valid_depth_mask_sampled = valid_depth_mask_flatten[..., flatten_index]  # [bs, sampled h * sampled w]

        depth_gt_reshaped = depth_gt_sampled.reshape(B, sample_h, sample_w)  # [bs, sampled h, sampled w]
        valid_depth_mask_reshaped = valid_depth_mask_sampled.reshape(B, sample_h, sample_w)  # [bs, sampled h, sampled w]

        # downsample and mask point cloud ground truth
        pts_world_gt = metas[0]['pts_world_gt']  # [bs, sampled h * sampled w, 3]
        # convert_data.convert_da_pts_nocolor(pts_world_gt.squeeze(0), 'gt_pts_world')
        pts_world_gt_sampled = pts_world_gt[:, flatten_index][valid_depth_mask_sampled]  # [1, n, 3]
        da_down_world_sampled = da_down_world[valid_depth_mask_sampled]
        # convert_data.convert_da_pts_nocolor(pts_world_gt_sampled.squeeze(0), 'gt_pts_down')

        output_dict = {
            'da_depth_input': pred,  # [1, h, w]
            'da_depth_label': depth_gt_reshaped,  # [1, h, w]
            'da_pts_input': da_down_world_sampled,  # [1, n, 3]
            'da_pts_label': pts_world_gt_sampled,  # [1, n, 3]
            'valid_mask': valid_depth_mask_reshaped,  # [sampled h, sampled w]
        }

        self.depthbranch_count['depthbranch_toc'] = time.time() - start_depthbranch_toc

        return output_dict, feature_maps, conf_map, inference_feature, da_down_cam, da_down_world

    @staticmethod
    def supervise_toc(current_toc, last_toc=None, denominator=1, log_view=True):
        # print & count
        if log_view: print(f"{24 * '-'} inference time: {current_toc['depthbranch_toc']:.4f} {24 * '-'}")
        for key in current_toc.keys():
            if key == 'depthbranch_toc':
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
                        if log_view: print(f"{3 * ' '} {subkey} time: {time_toc:.6f}s")
                else:
                    if last_toc is not None:
                        if key in last_toc.keys():
                            last_toc[key] += current_toc[key]
                        else:
                            last_toc[key] = 0
                            last_toc[key] += current_toc[key]
                    time_toc = current_toc[key] / denominator
                    if log_view: print(f"::: {key} time: {time_toc:.6f}s")

        if last_toc is not None:
            return last_toc


"""
    pred, da_down_cam, da_down_world, flatten_index, inference_feature, conf_map = self.lifter(  # [1, h, w]
        metas,
        self.flag_depthanything_as_gt,
        depthnet_output,  # [1, h, w]
        feature_maps,
        depth_feat,  # [1, 1, 128, h, w]
        depth_conf,  # [1, 1, h, w]
        ELUT,
    )

    # downsample and mask depth ground truth
    depth_gt_flatten = metas[0]['depth_gt'].flatten()  # [h * w]
    valid_depth_mask_flatten = metas[0]['valid_depth_mask'].flatten()  # [h * w]
    depth_gt_sampled = depth_gt_flatten[flatten_index]  # [sampled h * sampled w]
    valid_depth_mask_sampled = valid_depth_mask_flatten[flatten_index]  # [sampled h * sampled w]
    depth_gt_reshaped = depth_gt_sampled.reshape(30, 40)  # [sampled h, sampled w]
    valid_depth_mask_reshaped = valid_depth_mask_sampled.reshape(30, 40)  # [sampled h, sampled w]
    # convert_data.convert_depth(depth_gt_reshaped, 'gt_depth_down')

    # downsample and mask point cloud ground truth
    pts_world_gt = metas[0]['pts_world_gt'].unsqueeze(0)  # [1, sampled h * sampled w, 3]
    # convert_data.convert_da_pts_nocolor(pts_world_gt.squeeze(0), 'gt_pts_world')
    pts_world_gt_sampled = pts_world_gt[:, flatten_index][:, valid_depth_mask_sampled]  # [1, n, 3]
    da_down_world_sampled = da_down_world[valid_depth_mask_sampled]
    # convert_data.convert_da_pts_nocolor(pts_world_gt_sampled.squeeze(0), 'gt_pts_down')

    output_dict = {
        # 'da_depth_input': pred,  # [1, h, w]
        # 'da_depth_label': metas[0]['depth_gt'].unsqueeze(0),  # [1, h, w]
        # 'da_pts_input': da_world[metas[0]['valid_pts_mask']].unsqueeze(0),  # [1, n, 3]
        # 'da_pts_label': metas[0]['pts_world_gt'].unsqueeze(0),  # [1, n, 3]
        'da_depth_input': pred,  # [1, h, w]
        'da_depth_label': depth_gt_reshaped.unsqueeze(0),  # [1, h, w]
        'da_pts_input': da_down_world_sampled.unsqueeze(0),  # [1, n, 3]
        'da_pts_label': pts_world_gt_sampled,  # [1, n, 3]
        'valid_mask': valid_depth_mask_reshaped,  # [sampled h, sampled w]
    }
"""
