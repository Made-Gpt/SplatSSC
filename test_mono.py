import os
import cv2
import torch
import torch.distributed as dist
import os, time, argparse, os.path as osp, numpy as np

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 1, 3

# torchrun --nproc_per_node=1 vis_mono.py

import open3d as o3d
from dataset.nyu_utils import world2pix
from utils.iou_eval import IOUEvalBatch
from utils.iou_as_iso import SSCMetrics
from utils.iou_as_iso_stage import SSCMetricsStage

from utils.loss_record import LossRecord
from utils.load_save_util import revise_ckpt, revise_ckpt_2

from tqdm import tqdm
from mmengine import Config
from mmengine.runner import set_random_seed
from mmengine.logging.logger import MMLogger

import warnings

warnings.filterwarnings("ignore")

import sys

# sys.path.append('/EmbodiedOcc')
# sys.path.append('/EmbodiedOcc/Depth-Anything-V2/metric_depth')


def pass_print(*args, **kwargs):
    pass


def is_main_process():
    if not dist.is_available():
        return True
    elif not dist.is_initialized():
        return True
    else:
        return dist.get_rank() == 0


def main(args):
    # global settings
    torch.backends.cudnn.benchmark = True

    # load config
    cfg = Config.fromfile(args.py_config)

    set_random_seed(cfg.seed)
    cfg.work_dir = args.work_dir
    max_num_epochs = cfg.max_epochs
    eval_freq = cfg.eval_freq
    print_freq = cfg.print_freq

    # init DDP
    distributed = True
    world_size = int(os.environ["WORLD_SIZE"])  # number of nodes
    rank = int(os.environ["RANK"])  # node id
    gpu = int(os.environ['LOCAL_RANK'])
    dist.init_process_group(
        backend="nccl", init_method=f"env://",
        world_size=world_size, rank=rank
    )
    dist.barrier()
    torch.cuda.set_device(gpu)

    if not is_main_process():
        import builtins
        builtins.print = pass_print

    # configure logger
    if is_main_process():
        os.makedirs(args.work_dir, exist_ok=True)
        cfg.dump(osp.join(args.work_dir, osp.basename(args.py_config)))

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.work_dir, f'{timestamp}.log')
    logger = MMLogger(name='indoor_nyu_eval', log_file=log_file, log_level='INFO')
    # logger.info(f'Config:\n{cfg.pretty_text}')

    # build model
    from model import build_model
    my_model = build_model(cfg.model)

    if cfg.flag_depthanything_as_gt:
        my_model.pretrained_model.requires_grad_(False)

    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    logger.info(f'Number of params: {n_parameters}')
    # logger.info(f'Model:\n{my_model}')
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', True)
        if cfg.get('track_running_stats', False):
            my_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(my_model)
            logger.info('converted sync bn.')
        ddp_model_module = torch.nn.parallel.DistributedDataParallel
        my_model = ddp_model_module(
            my_model.cuda(),
            device_ids=[gpu],
            find_unused_parameters=find_unused_parameters)
    else:
        my_model = my_model.cuda()
    print('done ddp model')

    # build dataloader
    from dataset import build_dataloader
    train_dataset_loader, val_dataset_loader = \
        build_dataloader(
            cfg.train_dataset_config,
            cfg.val_dataset_config,
            cfg.train_wrapper_config,
            cfg.val_wrapper_config,
            cfg.train_loader_config,
            cfg.val_loader_config,
            data_type="mono",
            dist=distributed,
        )

    from loss import GPD_LOSS
    loss_func = GPD_LOSS.build(cfg.loss).cuda()

    # CalMeanIou = SSCMetricsStage(n_classes=11) # FIXME -- 12: include empty; 11: exclude empty
    CalMeanIou = SSCMetrics(n_classes=12)

    # resume and load
    cfg.resume_from = ''
    if osp.exists(osp.join(args.work_dir, 'latest.pth')):
        cfg.resume_from = osp.join(args.work_dir, 'latest.pth')
    if args.resume_from:
        cfg.resume_from = args.resume_from

    print('resume from: ', cfg.resume_from)
    print('work dir: ', args.work_dir)

    if cfg.resume_from and osp.exists(cfg.resume_from):
        map_location = 'cpu'
        ckpt = torch.load(cfg.resume_from, map_location=map_location)
        print(my_model.load_state_dict(revise_ckpt(ckpt['state_dict']), strict=False))
        epoch = ckpt['epoch']
        if 'best_val_iou' in ckpt:
            best_val_iou = ckpt['best_val_iou']
        global_iter = ckpt['global_iter']
        print(f'successfully resumed from epoch {epoch}')
    elif cfg.load_from:
        ckpt = torch.load(cfg.load_from, map_location='cpu')
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        state_dict = revise_ckpt(state_dict)
        try:
            print(my_model.load_state_dict(state_dict, strict=False))
        except:
            state_dict = revise_ckpt_2(state_dict)
            print(my_model.load_state_dict(state_dict, strict=False))

    save_dir = os.path.join(args.work_dir, 'vis_occ')
    os.makedirs(save_dir, exist_ok=True)

    metas_tensor_keys_inv = \
        ['depth_gt_np_valid', 'depth_gt_np', 'name', 'cam2img', 'world2img',
         'rgb_path', 'depth_path', 'num_depth', 'occ_mask_valid', 'occ_mask_valid_fov',
         'img_shape', 'img_aug_matrix', 'cam_vox_range', 'pix_z']

    metas_squeeze = ['depth_gt_np', 'depth_gt_np_valid', 'img_depthbranch', 'rgb', 'label']

    my_model.eval()
    loss_record = LossRecord(loss_func=loss_func)
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

    efficiency_count = {}
    acc_occ_inference_toc = 0
    local_iter_num = 0
    with torch.no_grad():
        # for i_iter_val, data in enumerate(val_dataset_loader):
        for i_iter_val, data in tqdm(enumerate(val_dataset_loader), total=len(val_dataset_loader), desc="Evaluating", unit="batch"):
            start_data_load_toc = time.time()
            # data is a tuple: (imgs_batch, meta_batch_dict, labels_batch)
            (imgs, meta_cache, label) = data

            # Use non_blocking=True for faster async transfer (works with pin_memory=True in DataLoader)
            imgs = imgs.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            # Iterate through the metas dictionary and move only tensors to GPU
            for k, v in meta_cache.items():
                if isinstance(v, torch.Tensor):
                    meta_cache[k] = v.cuda(non_blocking=True)

            metas = [meta_cache]

            start_occ_inference_toc = time.time()
            # print(f"data load time: {(start_occ_inference_toc - start_data_load_toc):6f}s")

            my_model.module.scene_init(metas[0]['img_depthbranch'].device)  # init this scene

            with torch.cuda.amp.autocast(cfg.amp):
                result_dict = my_model(imgs=imgs, metas=metas, points=None, label=label, grad_frames=cfg.grad_frames, test_mode=True)
            occ_inference_toc = time.time() - start_occ_inference_toc
            acc_occ_inference_toc += occ_inference_toc
            local_iter_num += 1

            voxel_predict = result_dict['ce_input'].argmax(dim=1).long()  # [1, 60, 60, 36]
            voxel_label = result_dict['ce_label'].long()  # [1, 60, 60, 36]
            voxel_fov_mask = result_dict['fov_mask'].long()

            voxel_label[voxel_label == 0] = 255  # ignore
            voxel_label[voxel_label == 12] = 0  # empty

            voxel_predict = voxel_predict.cpu()
            voxel_fov_mask = voxel_fov_mask.cpu()
            voxel_label = voxel_label.cpu()

            # nonempty_mask = voxel_label != 12  # mask empty voxels (according to the label)
            # CalMeanIou.add_batch(voxel_predict, voxel_label, nonempty=nonempty_mask, fov_mask=voxel_fov_mask)
            # CalMeanIou.add_batch(voxel_predict, voxel_label, fov_mask=voxel_fov_mask)
            # CalMeanIou.add_batch(voxel_predict, voxel_label, nonsurface=voxel_fov_mask)
            CalMeanIou.add_batch(voxel_predict, voxel_label)

            efficiency_count = my_model.module.supervise_toc(my_model.module.efficiency_count, last_toc=efficiency_count, denominator=local_iter_num, log_view=False)

            if cfg.print_eval_by_freq:
                if i_iter_val % print_freq == 0 and is_main_process():
                    stats = CalMeanIou.get_stats()

                    info_sem_cls = stats["iou_ssc"]
                    info_sem = stats["iou_ssc_mean"]
                    info_geo = stats["iou"]

                    info_sem_cls_str = np.array2string(info_sem_cls, precision=3, separator=' ', formatter={'float_kind': lambda x: f"{x:.3f}"})
                    single_occ_inference_time = acc_occ_inference_toc / local_iter_num
                    # my_model.module.supervise_toc(my_model.module.efficiency_count, denominator=local_iter_num, log_view=True)

                    logger.info(f"Cost {single_occ_inference_time:.5f}s per frame. \n"
                                f"|   Current Val   |   iou_ssc_mean   |   iou_geo_mean   |\n"
                                f"| {15 * ' '} |    {info_sem:.8f}    |    {info_geo:.8f}    |\n"
                                f"|   iou_ssc: {info_sem_cls_str}")
                    CalMeanIou.print_confusion_matrix()

                    acc_occ_inference_toc = 0
                    local_iter_num = 0

                    # for key in efficiency_count.keys():
                    #     efficiency_count[key] = 0

                    CalMeanIou.reset()

        if not cfg.print_eval_by_freq:
            stats = CalMeanIou.get_stats()

            info_sem_cls = stats["iou_ssc"]
            info_sem = stats["iou_ssc_mean"]
            info_geo = stats["iou"]

            info_sem_cls_str = np.array2string(info_sem_cls, precision=3, separator=' ',
                                               formatter={'float_kind': lambda x: f"{x:.3f}"})
            single_occ_inference_time = acc_occ_inference_toc / local_iter_num

            logger.info(
                f"[EVAL] Iter {i_iter_val}/{len(val_dataset_loader)}, cost {single_occ_inference_time:.5f}s per frame. \n"
                f"|   Current Val   |   iou_ssc_mean   |   iou_geo_mean   |\n"
                f"| {15 * ' '} |    {info_sem:.8f}    |    {info_geo:.8f}    |\n"
                f"|   iou_ssc: {info_sem_cls_str}")
            CalMeanIou.print_confusion_matrix()

            my_model.module.supervise_toc(efficiency_count, denominator=local_iter_num, log_view=True)

            CalMeanIou.reset()


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/vis_mono_config.py')
    parser.add_argument('--work-dir', type=str, default='/home/wyq/WorkSpace/workdir/vis_mono')
    parser.add_argument('--resume-from', type=str, default='')
    parser.add_argument('--frame-idx', type=int, nargs='+', default=[0])

    args, _ = parser.parse_known_args()
    main(args)


