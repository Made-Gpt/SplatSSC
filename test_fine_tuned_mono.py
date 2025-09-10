import os
import cv2
import torch
import torch.distributed as dist
import os, time, argparse, os.path as osp, numpy as np

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from utils.metric_depth import DepthMetrics
from utils.load_save_util import revise_ckpt, revise_ckpt_2

from tqdm import tqdm
from mmengine import Config
from mmengine.runner import set_random_seed
from mmengine.logging.logger import MMLogger

import warnings

warnings.filterwarnings("ignore")


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
    logger.info(f'Config:\n{cfg.pretty_text}')

    # build model
    from model import build_model
    my_model = build_model(cfg.model)

    if cfg.flag_depthanything_as_gt:
        if hasattr(my_model, 'depthanything'):
            my_model.depthanything.requires_grad_(False)
        if hasattr(my_model, 'vggt'):
            my_model.vggt.requires_grad_(False)

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
            dist=distributed,
        )

    from loss import GPD_LOSS
    loss_func = GPD_LOSS.build(cfg.loss).cuda()
    CalMeanMetrics = DepthMetrics()

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

    my_model.eval()
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

    depthbranch_count = {}
    local_iter_num = 0
    with torch.no_grad():
        # for i_iter_val, data in enumerate(val_dataset_loader):
        for i_iter_val, data in tqdm(enumerate(val_dataset_loader), total=len(val_dataset_loader), desc="Evaluating", unit="batch"):

            (imgs, meta_cache, label) = data
            # Use non_blocking=True for faster async transfer (works with pin_memory=True in DataLoader)
            imgs = imgs.cuda(non_blocking=True)
            # label = label.cuda(non_blocking=True)
            # Iterate through the metas dictionary and move only tensors to GPU
            for k, v in meta_cache.items():
                if isinstance(v, torch.Tensor):
                    meta_cache[k] = v.cuda(non_blocking=True)
            metas = [meta_cache]
            my_model.module.scene_init(metas[0]['img_depthbranch'].device)  # init this scene

            with torch.cuda.amp.autocast(cfg.amp):
                result_dict, _, _, _, _, _ = my_model(imgs=imgs, metas=metas)
            my_model.module.supervise_toc(my_model.module.depthbranch_count, depthbranch_count, log_view=False)
            local_iter_num += 1

            CalMeanMetrics.add_batch(result_dict)

            if cfg.print_eval_by_freq:
                if i_iter_val % print_freq == 0 and is_main_process():
                    stats = CalMeanMetrics.get_stats()
                    info_delta1 = stats["δ1"]
                    info_arel = stats["A.Rel"]
                    info_rmse = stats["RMSE"]
                    info_cd_l1 = stats["CD-l1"]
                    info_dtu_acc = stats["DTU-Acc"]
                    info_dtu_comp = stats["DTU-Comp"]
                    info_dtu_overall = stats["DTU-Overall"]

                    logger.info(
                        f"[EVAL] Iter {i_iter_val}/{len(val_dataset_loader)}\n" 
                        f"|{'Current val':^15}|{'δ1↑':^8}|{'A.Rel↓':^8}|{'RMSE↓':^8}|{'CD-l1.↓':^8}|{'Acc.↓':^8}|{'Comp.↓':^8}|{'Overall.↓':^10}|\n"
                        f"|{' ':^15}|{info_delta1:^8.5f}|{info_arel:^8.5f}|{info_rmse:^8.5f}|{info_cd_l1:^8.5f}|{info_dtu_acc:^8.5f}|{info_dtu_comp:^8.5f}|{info_dtu_overall:^10.5f}|"
                    )

                    my_model.module.supervise_toc(depthbranch_count, denominator=local_iter_num, log_view=True)
                    local_iter_num = 0
                    for key in depthbranch_count.keys():
                        depthbranch_count[key] = 0
                    CalMeanMetrics.reset_metric()

        if not cfg.print_eval_by_freq:
            stats = CalMeanMetrics.get_stats()
            info_delta1 = stats["δ1"]
            info_arel = stats["A.Rel"]
            info_rmse = stats["RMSE"]
            info_cd_l1 = stats["CD-l1"]
            info_dtu_acc = stats["DTU-Acc"]
            info_dtu_comp = stats["DTU-Comp"]
            info_dtu_overall = stats["DTU-Overall"]

            logger.info(
                f"[EVAL] Iter {i_iter_val}/{len(val_dataset_loader)}\n"
                f"|{'Current val':^15}|{'δ1↑':^8}|{'A.Rel↓':^8}|{'RMSE↓':^8}|{'CD-l1.↓':^8}|{'Acc.↓':^8}|{'Comp.↓':^8}|{'Overall.↓':^10}|\n"
                f"|{' ':^15}|{info_delta1:^8.5f}|{info_arel:^8.5f}|{info_rmse:^8.5f}|{info_cd_l1:^8.5f}|{info_dtu_acc:^8.5f}|{info_dtu_comp:^8.5f}|{info_dtu_overall:^10.5f}|"
            )

            my_model.module.supervise_toc(depthbranch_count, denominator=local_iter_num, log_view=True)
            CalMeanMetrics.reset_metric()


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/vis_mono_config.py')
    parser.add_argument('--work-dir', type=str, default='/home/wyq/WorkSpace/workdir/vis_mono')
    parser.add_argument('--resume-from', type=str, default='')
    parser.add_argument('--frame-idx', type=int, nargs='+', default=[0])

    args, _ = parser.parse_known_args()
    main(args)

