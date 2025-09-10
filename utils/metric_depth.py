import torch
from math import sqrt

def point_cloud_pairwise_distances(p1: torch.Tensor, p2: torch.Tensor, p_norm: int = 2) -> torch.Tensor:
    # As per user's file, this function expects [B,N,C] and [B,M,C] for cdist.
    # The class methods will call this with unsqueezed tensors.
    # Handling for empty inputs to avoid cdist error if one is [B,0,C]
    if p1.shape[1] == 0 or p2.shape[1] == 0: # p1 is [B,N,C], p2 is [B,M,C]
        return torch.empty(p1.shape[0], p1.shape[1], p2.shape[1], device=p1.device, dtype=p1.dtype)
    return torch.cdist(p1.contiguous(), p2.contiguous(), p=p_norm)


class DepthMetrics:
    def __init__(self, thresh1: float=1.25, f1_l2_threshold: float = 0.05):
        self.f1_l2_threshold = f1_l2_threshold
        self.thresh1 = thresh1
        self.batch_num = 0
        self.eps = 1e-6

        # Depth Metrics Accumulators
        self.delta1 = 0
        self.delta2 = 0
        self.delta3 = 0
        self.abs_rel = 0
        self.rmse = 0

        # PCD Metrics Accumulators
        self.pcd_acc = 0.0
        self.pcd_recall = 0.0
        self.pcd_f1 = 0.0
        self.pcd_cd_l1 = 0.0
        self.dtu_acc = 0.0
        self.dtu_comp = 0.0

        self._sum_sq_diff = 0.0

    def reset_metric(self):
        # Depth Metrics Accumulators
        self.batch_num = 0
        self.delta1 = 0
        self.delta2 = 0
        self.delta3 = 0
        self.abs_rel = 0
        self.rmse = 0

        # PCD Metrics Accumulators
        self.pcd_acc = 0.0
        self.pcd_recall = 0.0
        self.pcd_f1 = 0.0
        self.pcd_cd_l1 = 0.0
        self.dtu_acc = 0.0
        self.dtu_comp = 0.0

        self._sum_sq_diff = 0.0

    def add_batch(self, result_dict):
        """ --- depth metric --- """
        pred_d = result_dict['da_depth_input']  # [B, H, W] or [1, H, W]
        gt_d = result_dict['da_depth_label']  # [B, H, W] or [1, H, W]

        valid_d_mask = gt_d > self.eps  # (gt != 0) & (pred != 0)  # [1, H, W]
        pred_d_valid = pred_d[valid_d_mask]
        gt_d_valid = gt_d[valid_d_mask]

        # calculate metric for single frame
        ratio = torch.max(pred_d_valid / (gt_d_valid + self.eps), gt_d_valid / (pred_d_valid + self.eps))
        loc_delta1 = (ratio < self.thresh1).float().mean().item()
        loc_delta2 = (ratio < self.thresh1 ** 2).float().mean().item()
        loc_delta3 = (ratio < self.thresh1 ** 3).float().mean().item()
        loc_abs_rel = torch.mean(torch.abs(pred_d_valid - gt_d_valid) / (gt_d_valid + self.eps)).item()
        loc_sum_sq_diff = ((pred_d_valid - gt_d_valid) ** 2).sum().item()

        # update
        self.delta1 += loc_delta1
        self.delta2 += loc_delta2
        self.delta3 += loc_delta3
        self.abs_rel += loc_abs_rel
        self._sum_sq_diff += loc_sum_sq_diff

        """ --- point cloud metric ---"""
        pred_p = result_dict['da_pts_input'].float().squeeze(0)  # [n, 3]
        gt_p = result_dict['da_pts_label'].float().squeeze(0)  # [n, 3]

        num_pred_item = pred_p.shape[0]
        num_gt_item = gt_p.shape[0]

        # CD-L1
        dists_l1_p2g = point_cloud_pairwise_distances(pred_p, gt_p, p_norm=1)  # [num_pred, num_gt]
        min_dist_p2g_l1, _ = torch.min(dists_l1_p2g, dim=1)
        dists_l1_g2p = point_cloud_pairwise_distances(gt_p, pred_p, p_norm=1)  # [num_gt, num_pred]
        min_dist_g2p_l1, _ = torch.min(dists_l1_g2p, dim=1)
        cd_l1_frame = (min_dist_p2g_l1.mean() + min_dist_g2p_l1.mean()).item()

        # F1 Score and Accuracy (Precision) using L2 distances
        dists_l2_p2g = point_cloud_pairwise_distances(pred_p, gt_p, p_norm=2)  # [num_pred, num_gt]
        min_l2_dist_p2g, _ = torch.min(dists_l2_p2g, dim=1)
        precision_hits = (min_l2_dist_p2g < self.f1_l2_threshold).float().sum().item()
        acc_frame = (precision_hits / (num_pred_item + self.eps))

        dists_l2_g2p = point_cloud_pairwise_distances(gt_p, pred_p, p_norm=2)  # [num_gt, num_pred]
        min_l2_dist_g2p, _ = torch.min(dists_l2_g2p, dim=1)
        recall_hits = (min_l2_dist_g2p < self.f1_l2_threshold).float().sum().item()
        recall_frame = (recall_hits / (num_gt_item + self.eps))

        f1_frame = 0.0
        if (acc_frame + recall_frame) > self.eps:
            f1_frame = 2 * (acc_frame * recall_frame) / (acc_frame + recall_frame)

        # update
        self.pcd_cd_l1 += cd_l1_frame
        self.pcd_acc += acc_frame
        self.pcd_recall += recall_frame
        self.pcd_f1 += f1_frame
        self.dtu_acc += min_l2_dist_p2g.mean().item()
        self.dtu_comp += min_l2_dist_g2p.mean().item()

        self.batch_num += 1

    def get_stats(self):
        assert self.batch_num != 0, 'should have at least one valid output'
        self.delta1 /= self.batch_num
        self.delta2 /= self.batch_num
        self.delta3 /= self.batch_num
        self.abs_rel /= self.batch_num
        self.rmse = sqrt(self._sum_sq_diff / self.batch_num)

        self.pcd_acc /= self.batch_num
        self.pcd_recall /= self.batch_num
        self.pcd_f1 /= self.batch_num
        self.pcd_cd_l1 /= self.batch_num

        dtu_acc = self.dtu_acc / self.batch_num
        dtu_comp = self.dtu_comp / self.batch_num
        dtu_overall = (dtu_acc + dtu_comp) / 2

        stats = {
            'δ1': self.delta1, 'A.Rel': self.abs_rel, 'RMSE': self.rmse,

            f"Acc@{self.f1_l2_threshold:.3f}":self.pcd_acc,
            f"F1@{self.f1_l2_threshold:.3f}":self.pcd_f1,
            f"CD-l1":self.pcd_cd_l1,

            f"DTU-Acc": dtu_acc,
            f"DTU-Comp": dtu_comp,
            f"DTU-Overall": dtu_overall,
        }

        return stats


