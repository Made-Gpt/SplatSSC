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
max_epochs = 10
grad_max_norm = 35
amp = False
norm_map = True
grad_frames = None
flag_depthbranch = True  # FIXME !!! ours: True
use_camera_embed = False
track_running_stats = True
find_unused_parameters = True
print_eval_by_freq = False
flag_depthanything_as_gt = True

empty_idx = 12  # 0 ignore, 1~11 objects, 12 empty
semantic_dim = 11  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
ignore_label = 0

lut_size = ['r128', 'r128', 'r64']  # build KeyLUT  # FIXME
lut_range = [-10.24, -10.24, -5.0, 10.24, 10.24, 3.0]  # z-order encode range
pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
sample_shape = [30, 40]  # FIXME, baseline: [30, 40]
scale_range = [0.01, 0.16]  # FIXME, baseline: [0.01, 0.08]
image_size = (480, 640)
resize_lim = [1.0, 1.0]
num_frames = 1
offset = 0

_dim_ = 96
num_cams = 1
num_heads = 3
num_levels = 4
num_ref_pts = 1  # FIXME
num_anchor = 16200  # FIXME
num_anchor_init = 8100
num_cross_layer = 3
num_self_layer = 3
num_decoder = 3

stage_tg = 'main'
prob_loss_type = 'geo_3'  # FIXME !!! ours: 'geo_3'
semantics_activation = 'softmax'  # FIXME, 'softmax' for probability, 'identity' for logits
aggregate_mode = 'local_aggregate_prob_new'  # FIXME !!! ours: 'local_aggregate_prob_new'

path_to_your = "path/to/your"  # path/to/your
baseline_name = "EmbodiedOcc"  # SplatSSC  # EmbodiedOcc

vggt_pt = f"{path_to_your}/ckpt/VGGT-1B.pt"
da_pt = f"{path_to_your}/ckpt/depth_anything_v2_vitb.pth"
da_ft_pt = f"{path_to_your}/ckpt/finetune_scannet_depthanythingv2.pth"

load_from = f"{path_to_your}/{baseline_name}/result/occscannet/mini/main/latest.pth"
fine_tuning_ckpt = f"{path_to_your}/{baseline_name}/result/occscannet/depth/ftdav2/latest.pth"
zero_shot_ckpt = da_ft_pt

data_path = f"{path_to_your}/{baseline_name}/data/occscannet"  # data path
save_da = f"{path_to_your}/{baseline_name}/vis/occscannet/base/vis_occ"  # for visualization

# for debug
supervision = dict(
    head=False,
    lifter=False,
    encoder=False,
    segmentor=False,
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

ffn=dict(
    type='FFN',
    num_fcs=2,
    embed_dims=_dim_,
    feedforward_channels=_dim_*2,
    act_cfg=dict(type='ReLU', inplace=True),
)

spconv_layer=dict(
    type='SparseConv3D',
    save_da=save_da,
    norm_map=norm_map,
    in_channels=_dim_,
    embed_channels=_dim_,
    pc_range=pc_range,
    grid_size=[0.08] * 3,
    kernel_size=3,
    residual=True,  # FIXME, baseline: False
)

refine_layer = dict(
    type='SparseGaussian3DRefinementModule',
    save_da=save_da,
    norm_map=norm_map,  # FIXME, baseline: False
    embed_dims=_dim_,
    pc_range=pc_range,
    scale_range=scale_range,
    simple_mlp=True,  # FIXME, baseline: False
    share_split=False,
    restrict_xyz=True,
    restrict_scale=True,
    activate_scale='normalize',
    refine_scale=False,
    refine_rot=False,
    unit_xyz=[0.1, 0.1, 0.06],
    semantic_dim=semantic_dim,
)

ms_feature_fusion_layer = dict(
    type='MultiModuleFeatureDepthFusion',
    norm_map=norm_map,
    save_da=save_da,
    q_dims=64,
    kv_dims=_dim_,
    embed_dims=_dim_,
    num_groups=num_heads,
    num_levels=num_levels,
    attn_drop=0.15,
    image_size=image_size,
    use_camera_embed=use_camera_embed,
    use_grid_sample=False,  # FIXME, baseline: False
    use_conf_fuse=True,
    residual_mode="add",
    attn_mode="add_mlp"  # FIXME, chooses: 'dot_product', 'mlp', 'add_mlp', 'mlp_mlp'
)

model = dict(
    type='GaussianSegmentor',
    flag_depthbranch=flag_depthbranch,
    zero_shot_ckpt=zero_shot_ckpt,
    coding_param=coding_param,
    supervision=supervision['segmentor'],
    stage_tag="main",
    save_da=save_da,  # for debug and visualization
    backbone=None,
    neck=None,
    lifter=dict(
        type='GaussianLifter',
        save_da=save_da,
        embed_dims=_dim_,
        scale_mode='tanh',
        sample_mode='nearest',
        image_size=image_size,
        sample_shape=sample_shape,
        semantic_dim=semantic_dim,
        coding_param=coding_param,
        fine_tune_mode='map',
        supervision=supervision['lifter'],
        norm_layer=dict(type="LN", normalized_shape=_dim_),
        ms_feat_fuse_layer=ms_feature_fusion_layer,
        operation_order=[
            "fuse",
            "ffn",
            "norm",
        ],
        ffn=ffn,
    ),
    encoder=dict(
        type='SparseGaussianFormer',
        supervision=supervision['encoder'],
        coding_param=coding_param,
        num_ref_pts=num_ref_pts,
        lut_range=lut_range,
        ffn=ffn,
        norm_layer=dict(type="LN", normalized_shape=_dim_),
        anchor_encoder=dict(
            type='SparseGaussian3DEncoder',
            embed_dims=_dim_,
            semantic_dim=semantic_dim,  # FIXME, baseline: cls_dims-1
        ),
        deformable_model=dict(
            type='DeformableFeatureAggregation',
            norm_map=norm_map,
            save_da=save_da,
            embed_dims=_dim_,
            num_groups=num_heads,
            num_levels=num_levels,
            num_cams=num_cams,
            attn_drop=0.15,
            kps_generator=dict(
                type="SparseGaussian3DKeyPointsGenerator",
                norm_map=norm_map,
                save_da=save_da,
                num_learnable_pts=0,
                adaptive=True,  # FIXME, baseline: False
                fix_scale=[
                    [0, 0, 0],
                    [0.45, 0, 0],
                    [-0.45, 0, 0],
                    [0, 0.45, 0],
                    [0, -0.45, 0],
                    [0, 0, 0.45],
                    [0, 0, -0.45],
                ],
                pc_range=pc_range,
                scale_range=scale_range),
            use_deformable_func=True,
            use_camera_embed=use_camera_embed,
            residual_mode="add",
        ),
        num_decoder=num_decoder,
        refine_layer=refine_layer,
        spconv_layer=spconv_layer,
        operation_order=[
            "spconv",
            "norm",
            "deformable",
            "norm",
            "ffn",
            "norm",
            "refine",
        ] + [
            "spconv",
            "norm",
            "deformable",
            "norm",
            "ffn",
            "norm",
            "refine"
        ] + [
            "spconv",
            "norm",
            "deformable",
            "norm",
            "ffn",
            "norm",
            "refine"
        ]),
    head=dict(
        type='GaussianOccHead',
        save_da=save_da,
        norm_map=norm_map,
        empty_idx=empty_idx,
        ignore_label=ignore_label,
        num_classes=semantic_dim,
        cuda_kwargs=dict(
            scale_multiplier=2,  # FIXME, baseline: 3
            H=60, W=60, D=36,
            pc_min=[-51.2, -51.2, -5.0],
            scale_range=scale_range,
            grid_size=0.08),
        pc_range=pc_range,
        scale_range=scale_range,
        opas_thresh=1,  # consider opacity
        prob_loss_type=prob_loss_type,
        aggregate_mode=aggregate_mode,
        coordinate_boundary='clamp',
        semantics_activation=semantics_activation
    )
)

loss = dict(
    type='MultiLoss',
    loss_cfgs=[
        dict(
            type='FocalLoss',
            weight=100.0,
            gamma=2.5,  # FIXME, baseline: 2.0
            alpha=0.35,  # FIXME, baseline: 0.25
            cls_freq=[5080655412, 722756, 44793226, 41084591, 3416464, 21897101, 10609339, 13846320, 23470172, 263393,
                      30949122, 9871618],  # '3196722886' for ignore
            empty_idx=empty_idx,  # 12
            ignore_label=ignore_label,  # 0
            median_weight=False,  # FIXME, ours: False
            ignore_empty=False,  # FIXME, ours: False
            use_label_map=True,
            use_softmax=False,  # FIXME, ours: False
            use_custom=True,  # FIXME, ours: True
            input_dict={
                'pred': 'ce_input',
                'label': 'ce_label',
                'fov_mask': 'fov_mask'}),
        dict(
            type='LovaszLoss',
            weight=2.0,
            empty_idx=empty_idx,  # 12
            ignore_label=ignore_label,  # 0
            use_softmax=False,
            use_label_map=True,
            input_dict={
                'lovasz_input': 'ce_input',
                'lovasz_label': 'ce_label',
                'fov_mask': 'fov_mask'}),
        dict(
            type='Prob_Scal_Loss',
            weight=0.5,
            radius=0.16,
            loss_num=num_decoder,
            loss_list=[0, 1, 2],  # 3
            empty_idx=empty_idx,  # 12
            ignore_label=ignore_label,  # 0
            equal_weight=False,  # FIXME, ours: False
            cuda_kwargs=dict(
                scale_multiplier=2,  # FIXME, baseline: 3
                H=60, W=60, D=36,
                pc_min=[-51.2, -51.2, -5.0],
                grid_size=0.08),
            input_dict={
                'ssc_target': 'ce_label',
                'gaussian_cache': 'gaussian_cache',
                'cov_inv_cache': 'cov_inv_cache',
                'sampled_xyz': 'sampled_xyz',
                'fov_mask': 'fov_mask',
                'pc_min': 'pc_min',
            }),
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
    data_tg='mini', # 'base' for base-set
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
    data_tg='mini',  # 'base' for base-set
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

