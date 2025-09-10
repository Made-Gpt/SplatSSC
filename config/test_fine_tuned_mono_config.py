optimizer_wrapper = dict(
    optimizer = dict(
        type='AdamW',
        lr=2e-4,
        weight_decay=0.01,
    ),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1)}
    ),
)

seed = 1
eval_freq = 1
print_freq = 50
max_epochs = 20
grad_max_norm = 35
amp = False
grad_frames = None
flag_depthbranch = True  # FIXME, baseline: True
track_running_stats = True
find_unused_parameters = True
flag_depthanything_as_gt = True
print_eval_by_freq = False

# ignore_label = [0, 12]  # ['ignore', 'empty']
ignore_label = 0
semantic_dim = 11
empty_idx = 12  # 0 ignore, 1~11 objects, 12 empty
cls_dims = 13  # 12: exclude 'empty' label, 13: include empty label

lut_size = ['r64', 'r64', 'r128']  # build KeyLUT
pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
sample_shape = [30, 40]
scale_range = [0.01, 0.08]  # [0.01, 0.08]
image_size = [480, 640]
resize_lim = [1.0, 1.0]
num_frames = 1
offset = 0

_dim_ = 96
num_cams = 1
num_heads = 3  # num_groups
num_levels = 4
num_off_pts = 3  # FIXME
num_ref_pts = 1  # FIXME
num_self_layer = 3
num_cross_layer = 3
num_anchor_init = 8100
num_decoder_fillhead = 2
semantics_activation = 'identity'
use_camera_embed = False

stage_tg = 'depth_branch'
path_to_your = "path/to/your"  # path/to/your
baseline_name = "EmbodiedOcc"  # SplatSSC  # EmbodiedOcc

save_da = f"{path_to_your}/{baseline_name}/vis/occscannet/depth/vis_occ_da"
vggt_pt = f"{path_to_your}/ckpt/VGGT-1B.pt"
da_pt = f"{path_to_your}/ckpt/depth_anything_v2_vitb.pth"
da_ft_pt = f"{path_to_your}/ckpt/finetune_scannet_depthanythingv2.pth"
# fine_tuning_ckpt = f"{path_to_your}/{baseline_name}/result/occscannet/depth/ftdav2/latest.pth"
fine_tuning_ckpt = f"{path_to_your}/{baseline_name}/result/occscannet/depth/dav2/latest.pth"

data_path = f"{path_to_your}/{baseline_name}/data/occscannet"
load_from = fine_tuning_ckpt
zero_shot_ckpt = da_ft_pt  # da_pt

# for debug
supervision = dict(
    head=False,
    lifter=False,
    encoder=False,
    segmentor=False,
    self_encoder=False,
    serialization=False,
)

# config KeyLUT and anchor position coding
coding_param = dict(
    depth=16,
    order='z',
    lut_size=lut_size,
    shuffle_orders=False,
    log_view=supervision['serialization']
)

ms_feature_fusion_layer = dict(
    type='MultiModuleFeatureDepthFusion',
    norm_map=True,
    save_da=save_da,
    q_dims=64,  # FIXME, baseline: _dim_(96)
    kv_dims=_dim_,
    embed_dims=_dim_,
    num_groups=num_heads,
    num_levels=num_levels,
    attn_drop=0.15,
    use_camera_embed=use_camera_embed,
    use_grid_sample=False,  # FIXME, baseline: False
    use_conf_fuse=True,
    residual_mode="add",
    attn_mode="add_mlp"  # FIXME, baseline: 'mlp', chooses: 'dot_product', 'add_mlp', 'mlp_mlp'
)

model = dict(
    type='FineTuneDepthBranch',
    supervision=supervision['segmentor'],
    flag_depthanything_as_gt=False,
    coding_param=coding_param,
    zero_shot_ckpt=zero_shot_ckpt,
    save_da=save_da,  # for debug and visualization
    backbone=None,
    neck=None,
    lifter=dict(
        type='DepthBranchLifter',
        save_da=save_da,
        embed_dims=_dim_,
        scale_mode='tanh',  # FIXME, baseline: 'ss_sigmoid'
        sample_mode='nearest',  # FIXME, baseline: 'z-order'
        sample_shape=sample_shape,
        semantic_dim=semantic_dim,  # FIXME, baseline: cls_dim-1
        coding_param=coding_param,
        fine_tune_mode='map',  # FIXME, baseline: 'scale'
        supervision=supervision['lifter'],
        flag_depthbranch=flag_depthbranch,
        ffn=dict(
            type='FFN',
            num_fcs=2,
            embed_dims=_dim_,
            feedforward_channels=_dim_*2,
            act_cfg=dict(type='ReLU', inplace=True),
        ),
        norm_layer=dict(type="LN", normalized_shape=_dim_),
        ms_feat_fuse_layer=ms_feature_fusion_layer,
        operation_order = [
            "fuse",
            "ffn",
            "norm",
        ]
    ),
)

loss = dict(
    type='MultiLoss',
    loss_cfgs=[
        dict(
            type='Depth_Huber_Loss',
            weight=10,  # 1.0
            input_dict={
                'depth_preds': 'da_depth_input',
                'depth_labels': 'da_depth_label'
            }
        ),
        dict(
            type='PCD_Huber_Loss',
            weight=20,  # 1.0
            input_dict={
                'point_preds': 'da_pts_input',
                'point_labels': 'da_pts_label'
            }
        ),
        dict(
            type='Depth_Gradient_Loss',
            weight=0.8,  # 1.0
            input_dict={
                'depth_preds': 'da_depth_input',  # [B, H, W]
                'depth_labels': 'da_depth_label',  # [B, H, W]
                'valid_mask': 'valid_mask',  # [H, W]
            },
            loss_type='l1',
        )
    ]
)

train_dataset_config = dict(
    type='Scannet_Scene_OpenOccupancy_Dataset',
    data_path = data_path,
    num_frames = num_frames,
    offset = offset,
    empty_idx = empty_idx,
    phase='train',
    num_pts=num_anchor_init,
    data_tg='base', # 'mini' for mini-set
    stage_tg=stage_tg,
)

val_dataset_config = dict(
    type='Scannet_Scene_OpenOccupancy_Dataset',
    data_path = data_path,
    num_frames = num_frames,
    offset = offset,
    empty_idx=empty_idx,
    phase='test',
    num_pts=num_anchor_init,
    data_tg='mini', # 'mini' for mini-set
    stage_tg=stage_tg,
)

train_wrapper_config = dict(
    type='Scannet_Scene_Occ_DatasetWrapper',
    final_dim = [480, 640], 
    resize_lim = resize_lim,
    phase='train', 
)

val_wrapper_config = dict(
    type='Scannet_Scene_Occ_DatasetWrapper',
    final_dim = [480, 640],
    resize_lim = resize_lim,
    phase='test', 
)

train_loader_config = dict(
    batch_size = 1,
    shuffle = True,
    num_workers = 8,
)

val_loader_config = dict(
    batch_size = 1,
    shuffle = False,
    num_workers = 2,
)
