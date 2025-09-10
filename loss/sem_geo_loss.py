import torch
import torch.nn.functional as F
from sympy.integrals.risch import NonElementaryIntegral

from loss.ops.occ_prob import OccAggregator
from .base_loss import BaseLoss
from . import GPD_LOSS


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


@GPD_LOSS.register_module()
class Geo_Scal_Loss(BaseLoss):

    def __init__(self, weight=1.0, ignore_label=0, use_softmax=True,
                 empty_idx=12, input_dict=None, prs_weight=None, **kwargs):
        super().__init__(weight)

        if input_dict is None:
            self.input_dict = {
                'pred': 'ce_input',
                'ssc_target': 'ce_label'
            }
        else:
            self.input_dict = input_dict

        if prs_weight is None:
            self.prs_weight = [1, 1, 1]
        else:
            self.prs_weight = prs_weight

        self.loss_func = self.geo_scal_loss
        self.use_softmax = use_softmax
        self.ignore_label = ignore_label
        self.empty_idx=empty_idx

    def geo_scal_loss(self, pred, ssc_target, fov_mask=None):
        pred = pred.float()  # pred result, [bs, 12, 60, 60, 36]
        ssc_target = ssc_target.long()  # ground truth, [bs, 60, 60, 36]
        # Get softmax probabilities
        if self.use_softmax:
            pred = F.softmax(pred, dim=1)

        # use the max probability among the 11 classes as an indicator of "non-empty" prediction strength
        empty_probs = pred[:, self.ignore_label]  # pred have only 12 classes. '0' -- ignore, '1'~'11' -- valid
        nonempty_probs = 1 - empty_probs  # [bs, 60, 60, 36]

        mask = ssc_target != self.ignore_label  # 'ignore' area: 0, 'empty' + 'valid' area: 1
        non_empty_mask = ssc_target != self.empty_idx  # 'empty' area: 0, 'ignore' + 'valid' area: 1
        if fov_mask is not None:
            mask = mask & fov_mask

        # mask gt, [0, 1]
        valid_target = non_empty_mask[mask].float()  # get non-ignore area, where 'empty' area: 0, 'valid' area: 1
        valid_probs = nonempty_probs[mask]  # get non-ignore area
        spec_probs = empty_probs[mask]  # get non-ignore area, where invalid area has higher probability

        eps = 1e-5
        # valid voxels supervision
        intersection = (valid_target * valid_probs).sum()  # gt non-empty labels
        precision = intersection / (valid_probs.sum()+eps)  # (nonempty_probs.sum()+eps) is predicted non-empty voxels
        recall = intersection / (valid_target.sum()+eps)
        spec_target = 1 - valid_target  # where 'empty' area: 1, 'valid' area: 0
        spec = (spec_target * spec_probs).sum() / (spec_target.sum()+eps)

        loss = (
            F.binary_cross_entropy(precision, torch.ones_like(precision))  # precision
            + F.binary_cross_entropy(recall, torch.ones_like(recall))  # recall
            + F.binary_cross_entropy(spec, torch.ones_like(spec))  # recall (spec)
        )

        return loss


@GPD_LOSS.register_module()
class Sem_Scal_Loss(BaseLoss):  # Using BaseLoss as in your original file

    def __init__(self, weight=1.0, ignore_label=0, empty_idx=12, prs_weight=None,
                 use_softmax=True, sem_cls_range=None, input_dict=None, **kwargs):
        # THIS __INIT__ METHOD IS UNCHANGED AS PER YOUR REQUIREMENT
        super().__init__(weight)

        if input_dict is None:
            self.input_dict = {
                'pred': 'ce_input',
                'ssc_target': 'ce_label'
            }
        else:
            self.input_dict = input_dict

        if prs_weight is None:
            self.prs_weight = [1, 1, 1]
        else:
            self.prs_weight = prs_weight

        self.loss_func = self.sem_scal_loss
        self.sem_cls_range = sem_cls_range  # Expected to be [0, 11] for 11 valid classes
        self.use_softmax = use_softmax
        self.ignore_label = ignore_label
        self.empty_idx = empty_idx

    def sem_scal_loss(self, pred, ssc_target, fov_mask=None):
        # pred: Prediction from the model, shape [B, 12, H, W, D] (logits)
        # ssc_target: Ground truth, shape [B, H, W, D] (remapped: 0-10 for valid, 11 for background/empty, self.ignore_label for true ignore)
        # fov_mask: FOV mask, shape [H, W, D]

        pred = pred.float()
        ssc_target = ssc_target.long()

        # Get softmax probabilities across the 12 channels
        if self.use_softmax:
            pred = F.softmax(pred, dim=1)  # 'pred' now holds probabilities [B, 12, H, W, D]
        loss = 0.0  # Accumulator for the sum of losses for each valid class
        count = 0.0  # Counter for valid classes that have GT instances in the current batch/mask
        non_ignore_mask = (ssc_target != self.ignore_label).squeeze(0)  # 'ignore' area: 0, 'empty' + 'valid' area: 1
        # non_empty_mask = (ssc_target != self.empty_idx).squeeze(0)  # 'empty' area: 0, 'ignore' + 'valid' area: 1

        if fov_mask is not None:
            # mask = (non_ignore_mask & non_empty_mask & fov_mask).unsqueeze(0)  # Add batch dim
            # mask = (non_ignore_mask & fov_mask).unsqueeze(0)  # Add batch dim
            mask = (non_ignore_mask & fov_mask)  # Add batch dim
        else:
            # mask = (non_ignore_mask & non_empty_mask).unsqueeze(0)  # Add batch dim
            # mask = non_ignore_mask.unsqueeze(0)  # Add batch dim
            mask = non_ignore_mask  # Add batch dim

        # Loop through the 11 valid classes
        for i in range(self.sem_cls_range[0], self.sem_cls_range[1]):
            if i in [self.ignore_label, self.empty_idx]:
                continue

            p = pred[:, i]
            # (f"p: {p.shape}, pred shape: {pred.shape}, ssc_target shape: {ssc_target.shape}, mask shape: {mask.shape}")

            # Remove unknown voxels
            target_ori = ssc_target  # all labels
            p = p[mask]
            target = ssc_target[mask]  # valid labels

            # target voxels in valid labels
            completion_target = torch.ones_like(target)
            completion_target[target != i] = 0
            # target voxels in all labels
            completion_target_ori = torch.ones_like(target_ori).float()
            completion_target_ori[target_ori != i] = 0
            if torch.sum(completion_target) > 0:
                count += 1.0
                nominator = torch.sum(p * completion_target)
                loss_class = 0
                if torch.sum(p) > 0:
                    precision = nominator / (torch.sum(p))
                    loss_precision = F.binary_cross_entropy(
                        precision, torch.ones_like(precision)
                    )
                    loss_class += loss_precision * self.prs_weight[0]
                if torch.sum(completion_target) > 0:
                    recall = nominator / (torch.sum(completion_target))
                    loss_recall = F.binary_cross_entropy(recall, torch.ones_like(recall))
                    loss_class += loss_recall * self.prs_weight[1]
                if torch.sum(1 - completion_target) > 0:
                    specificity = torch.sum((1 - p) * (1 - completion_target)) / (
                        torch.sum(1 - completion_target)
                    )
                    loss_specificity = F.binary_cross_entropy(
                        specificity, torch.ones_like(specificity)
                    )
                    loss_class += loss_specificity * self.prs_weight[2]
                loss += loss_class

        if count == 0:
            return torch.tensor(10.0, device=pred.device)
        else:
            return loss / count


@GPD_LOSS.register_module()
class Prob_Scal_Loss(BaseLoss):

    def __init__(self, weight=1.0, ignore_label=0, empty_idx=12, sem_cls_range=None, loss_list=None, input_dict=None, cuda_kwargs:dict=None,
                 radius=0.16, grid_shape=None, grid_size=0.08, max_clamp=15, equal_weight=False, prs_weight=None,
                 use_ignore=True, use_softmax=False, use_opas=True, loss_type='geo_3', base_loss='bce', **kwargs):
        super().__init__(weight)

        if input_dict is None:
            self.input_dict = {
                'ssc_target': 'ce_label',
                'sem_cache': 'sem_cache', 
                'gaussian_cache': 'gaussian_cache',
                'cov_inv_cache': 'cov_inv_cache',
                'sampled_xyz': 'sampled_xyz',
                'fov_mask': 'fov_mask',
                'pc_min': 'pc_min',
            }
        else:
            self.input_dict = input_dict

        if loss_list is None:
            self.loss_list = [0, 1, 2]
        else:
            self.loss_list = loss_list

        if sem_cls_range is None:
            self.sem_cls_range = [1, 11]
        else:
            self.sem_cls_range = sem_cls_range

        valid_radius = [0.08, 0.10, 0.12, 0.16, 0.20, 0.24, 0.32, 0.36, 0.40]
        offset_dict = {
            '0.08': [-1, 0, 1], '0.10': [-1, 0, 1], '0.12': [-1, 0, 1],
            '0.16': [-2, -1, 0, 1, -2], '0.20': [-2, -1, 0, 1, -2], '0.24': [-2, -1, 0, 1, -2],
            '0.32': None, '0.36': None, '0.40': None
        }
        self.radius = radius
        if radius in valid_radius:
            self.off_range = offset_dict[f'{radius:.2f}']
        else:
            self.off_range = [-2, -1, 0, 1, -2]

        if grid_shape is None:
            self.grid_shape = [60, 60, 36]
        else:
            self.grid_shape = grid_shape
        self.grid_size = grid_size

        if prs_weight is None:
            self.prs_weight = [1, 1, 1]
        else:
            self.prs_weight = prs_weight

        if cuda_kwargs is not None:
            self.occ_aggregator = OccAggregator(**cuda_kwargs)

        self.loss_func = self.prob_scal_loss
        loss_str = loss_type.split('_')  # eg, 'sem_3' --> ['sem', '3']
        self.sem_or_geo = loss_str[0]  # 'sem' or 'geo'
        self.loss_num = int(loss_str[1])  # 1 or 2 or 3
        self.ignore_label = ignore_label
        self.empty_idx=empty_idx
        self.max_clamp = max_clamp
        self.epsilon = 1e-6
        self.equal_weight = equal_weight
        self.use_softmax = use_softmax
        self.use_ignore = use_ignore
        self.use_opas = use_opas
        self.base_loss = base_loss  # "bce" or "mse"
        self.cuda_kwargs = cuda_kwargs

    def prob_scal_loss(self, ssc_target, sem_cache, gaussian_cache, cov_inv_cache, sampled_xyz, pc_min, fov_mask=None):
        ssc_target = ssc_target.long()  # ground truth, [bs, 60, 60, 36]
        ssc_bin = ((ssc_target != self.ignore_label) & (ssc_target != self.empty_idx)).to(torch.uint8)  # [12, 0] --> 0, [else] --> 1.

        if self.use_ignore:
            mask = ssc_target != self.ignore_label  # 'ignore' area: 0, 'empty' + 'valid' area: 1
        else:
            mask = ssc_target != -1
        if fov_mask is not None:
            mask = mask & fov_mask
        # mask: [bs, 60, 60, 36]

        loss = 0.0
        for i in self.loss_list:  # [0, 1, 2]
            if self.sem_or_geo == 'geo':
                local_loss = self.geo_scale_loss(ssc_bin, gaussian_cache[i], cov_inv_cache[i], sampled_xyz, pc_min, mask, i)
            elif self.sem_or_geo == 'sem':
                pred = sem_cache[i].float()
                local_loss = self.sem_scale_loss(pred, ssc_target, mask)
            elif self.sem_or_geo == 'sem-geo':
                local_loss = self.sem_geo_loss(ssc_target, gaussian_cache[i], cov_inv_cache[i], sampled_xyz, pc_min, mask)
            else:
                raise ValueError(f"Invalid Loss Type {self.sem_or_geo}")

            if self.equal_weight:
                loss += local_loss
            else:
                if i < self.loss_num-1:
                    loc_w = (i + 1) / self.loss_num * 0.5
                else:
                    loc_w = self.loss_num
                loss += local_loss * loc_w

        return loss

    def geo_scale_loss(self, ssc_bin, gaussian, cov_inv, sampled_xyz, pc_min, mask, layer_idx):
        bs = ssc_bin.shape[0]
        if self.cuda_kwargs is not None:
            means = gaussian.means
            scales = gaussian.scales
            if self.use_opas:
                opacities = gaussian.opacities.flatten(1, 2)
            else:
                opacities = torch.ones((bs, means.shape[1]), dtype=means.dtype, device=means.device)
            # occ_prob = self.occ_aggregator(
            #     sampled_xyz,
            #     means,
            #     scales,
            #     cov_inv,
            #     opacities,
            #     pc_min,
            # )  # 129600
            occupancies = []
            for i in range(len(sampled_xyz)):  # batch size
                loc_occ = self.occ_aggregator(
                    sampled_xyz[i:(i+1)],
                    means[i:(i+1)],
                    scales[i:(i+1)],
                    cov_inv[i:(i+1)],
                    opacities[i:(i+1)],
                    pc_min[i:(i+1)],
                )  # 129600
                occupancies.append(loc_occ)
            occupancies = torch.stack(occupancies, dim=0)  # [bs, 129600]
            occ_prob = occupancies.reshape(bs, 60, 60, 36)  # [bs, 60, 60, 36]
        else:

            pred_probs = gaussian.means.float()
            occ_prob = self.xyz2occ(pred_probs.squeeze(0), pc_min)
        occ_prob = torch.clamp(occ_prob, self.epsilon, 1 - self.epsilon)  # [bs, 60, 60, 36]

        valid_target = ssc_bin[mask].float()  # [num_valid]
        valid_probs = occ_prob[mask]  # [num_valid]
        spec_probs = (1 - occ_prob)[mask]  # [num_valid]

        intersection = (valid_target * valid_probs).sum()  # gt non-empty labels
        # valid voxels supervision
        precision = intersection / (valid_probs.sum() + self.epsilon)  # (nonempty_probs.sum()+eps) is predicted non-empty voxels
        recall = intersection / (valid_target.sum() + self.epsilon)
        # specificity voxels supervision
        spec_target = 1 - valid_target  # where 'empty' area: 1, 'valid' area: 0
        spec = (spec_target * spec_probs).sum() / (spec_target.sum() + self.epsilon)

        loss_precision = F.binary_cross_entropy(precision, torch.ones_like(precision))  # precision
        loss_recall = F.binary_cross_entropy(recall, torch.ones_like(recall))  # recall
        loss_spec = F.binary_cross_entropy(spec, torch.ones_like(spec))  # recall (spec)
        if isinstance(self.prs_weight[layer_idx], list):
            loss = (loss_precision * self.prs_weight[layer_idx][0] +
                    loss_recall * self.prs_weight[layer_idx][1] +
                    loss_spec * self.prs_weight[layer_idx][2])
        else:
            loss = loss_precision * self.prs_weight[0] + loss_recall * self.prs_weight[1] + loss_spec * self.prs_weight[2]
        return loss

    def sem_scale_loss(self, pred, ssc_target, mask):
        # Get softmax probabilities across the 12 channels
        if self.use_softmax:
            pred = F.softmax(pred, dim=1)  # 'pred' now holds probabilities [B, 12, H, W, D]
        loss = 0.0  # Accumulator for the sum of losses for each valid class
        count = 0.0  # Counter for valid classes that have GT instances in the current batch/mask

        # Loop through the 11 valid classes
        for i in range(self.sem_cls_range[0], self.sem_cls_range[1]):
            # safe operation
            if i in [self.ignore_label, self.empty_idx]:
                continue

            p = pred[:, i]
            target_ori = ssc_target  # all labels
            p = p[mask]
            target = ssc_target[mask]  # valid labels

            # target voxels in valid labels
            completion_target = torch.ones_like(target)
            completion_target[target != i] = 0
            # target voxels in all labels
            completion_target_ori = torch.ones_like(target_ori).float()
            completion_target_ori[target_ori != i] = 0
            if torch.sum(completion_target) > 0:
                count += 1.0
                nominator = torch.sum(p * completion_target)
                loss_class = 0
                # precise
                if torch.sum(p) > 0:
                    precision = nominator / (torch.sum(p))
                    loss_precision = F.binary_cross_entropy(
                        precision, torch.ones_like(precision)
                    )
                    loss_class += loss_precision
                # recall
                if torch.sum(completion_target) > 0:
                    recall = nominator / (torch.sum(completion_target))
                    loss_recall = F.binary_cross_entropy(recall, torch.ones_like(recall))
                    loss_class += loss_recall
                # spec
                if torch.sum(1 - completion_target) > 0:
                    specificity = torch.sum((1 - p) * (1 - completion_target)) / (
                        torch.sum(1 - completion_target)
                    )
                    loss_specificity = F.binary_cross_entropy(
                        specificity, torch.ones_like(specificity)
                    )
                    loss_class += loss_specificity

                # bce loss:
                loss += loss_class

        if count == 0:
            return torch.tensor(10.0, device=pred.device)
        else:
            return loss / count

    def sem_geo_loss(self, ssc_target, gaussian, cov_inv, sampled_xyz, pc_min, mask):
        loss = 0.0  # total loss
        count = 0.0  # valid label number
        cls_indices = torch.argmax(gaussian.semantics.squeeze(0), dim=-1)

        # valid classes
        for class_id in range(self.sem_cls_range[0], self.sem_cls_range[1]):
            # safe operation,
            if class_id in [self.ignore_label, self.empty_idx]:
                continue

            # ground truth
            loc_gt_mask = (ssc_target == class_id)
            num_gt_voxels = torch.sum(loc_gt_mask)
            # loc gaussian
            loc_gaussian_mask = (cls_indices == (class_id-1))  # [num]
            num_gaussian = torch.sum(loc_gaussian_mask)
            if num_gt_voxels == 0 and num_gaussian == 0:
                continue

            count += 1.0
            if num_gaussian > 0:
                if self.cuda_kwargs is not None:
                    loc_means = gaussian.means[:, loc_gaussian_mask]  # [1, num, 3]
                    loc_scales = gaussian.scales[:, loc_gaussian_mask]  # [1, num, 3]
                    loc_cov_inv = cov_inv[:, loc_gaussian_mask]
                    if self.use_opas:
                        opacities = gaussian.opacities.flatten(1, 2)
                        loc_opas = opacities[:, loc_gaussian_mask]  # [1, num, 1]
                    else:
                        loc_opas = torch.ones((1, loc_means.shape[1]), dtype=loc_means.dtype, device=loc_means.device)
                    occ_prob = self.occ_aggregator(
                        sampled_xyz,
                        loc_means,
                        loc_scales,
                        loc_cov_inv,
                        loc_opas,
                        pc_min,
                    )  # 129600
                    occ_prob = occ_prob.reshape(60, 60, 36).unsqueeze(0)  # [1, 60, 60, 36]
                else:
                    pred_probs = gaussian.means.float()
                    occ_prob = self.xyz2occ(pred_probs.squeeze(0), pc_min)
                occ_prob = torch.clamp(occ_prob, self.epsilon, 1 - self.epsilon)
            else:
                occ_prob = torch.zeros_like(ssc_target, dtype=torch.float32)

            # calculate Precision, Recall, Specificity in target region
            valid_target = loc_gt_mask[mask].float()
            valid_probs = occ_prob[mask]

            intersection = (valid_target * valid_probs).sum()
            denominator = valid_probs.sum() + valid_target.sum()
            # precision = intersection / (valid_probs.sum() + self.epsilon)
            # recall = intersection / (valid_target.sum() + self.epsilon)

            spec_target = 1 - valid_target
            spec_probs = 1 - valid_probs
            specificity = (spec_target * spec_probs).sum() / (spec_target.sum() + self.epsilon)

            # loc_loss_precision = F.binary_cross_entropy(precision, torch.ones_like(precision))
            # loc_loss_recall = F.binary_cross_entropy(recall, torch.ones_like(recall))
            loc_loss_spec = F.binary_cross_entropy(specificity, torch.ones_like(specificity))
            # loc_loss_precision = torch.clamp(loc_loss_precision, max=self.max_clamp)
            # loc_loss_recall = torch.clamp(loc_loss_recall, max=self.max_clamp)
            loc_loss_spec = torch.clamp(loc_loss_spec, max=self.max_clamp)
            # loc_loss = loc_loss_precision + loc_loss_recall + loc_loss_spec

            loss_bce = F.binary_cross_entropy(valid_probs, valid_target, reduction='mean')
            loss_dice = 1.0 - (2.0 * intersection + self.epsilon) / (denominator + self.epsilon)
            loc_loss = loss_bce + loss_dice + loc_loss_spec

            loss += loc_loss

        if count > 0:
            return loss / count
        else:
            return torch.tensor(0.0, device=ssc_target.device)

    def xyz2occ(self, anchors_xyz, pc_min):
        # anchors_xyz: [num, 3]
        num_anchor = anchors_xyz.shape[0]
        device = anchors_xyz.device

        g_h, g_w, g_d = self.grid_shape  # Unpack grid shape
        if num_anchor == 0:
            return torch.zeros((1, g_h, g_w, g_d), device=device, dtype=torch.float32)

        # Convert anchor points to grid indices, detach to avoid gradient issues with integer conversion
        xyz_world_int = ((anchors_xyz - pc_min) / self.grid_size).detach().to(torch.long)  # Ensure int64 from start

        if self.off_range is None:
            # Use all grid indices
            x_indices = torch.arange(self.grid_shape[0], device=device, dtype=torch.long)
            y_indices = torch.arange(self.grid_shape[1], device=device, dtype=torch.long)
            z_indices = torch.arange(self.grid_shape[2], device=device, dtype=torch.long)

            # Create meshgrid and flatten to get all possible indices
            X, Y, Z = torch.meshgrid(x_indices, y_indices, z_indices, indexing='ij')
            valid_indices = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1)  # [grid_volume, 3]
        else:
            # Generate 3x3x3 neighborhood offsets
            offsets = torch.tensor([
                [i, j, k] for i in self.off_range for j in self.off_range for k in self.off_range
            ], device=device, dtype=torch.long)  # Use int64 from creation

            # Expand anchors and add offsets to get all neighbor indices
            neighbor_indices = xyz_world_int.unsqueeze(1) + offsets.unsqueeze(0)  # [num_anchor, 27, 3]
            neighbor_indices = neighbor_indices.reshape(-1, 3)  # [num_anchor * 27, 3]

            # Filter valid indices (within grid bounds)
            valid_mask = (
                    (neighbor_indices[:, 0] >= 0) & (neighbor_indices[:, 0] < g_h) &
                    (neighbor_indices[:, 1] >= 0) & (neighbor_indices[:, 1] < g_w) &
                    (neighbor_indices[:, 2] >= 0) & (neighbor_indices[:, 2] < g_d)
            )
            valid_indices = neighbor_indices[valid_mask]  # [num_valid, 3]

        # Remove duplicates using torch.unique
        valid_indices_unique = torch.unique(valid_indices, dim=0)  # [num_unique_valid, 3] - already int64

        # Convert indices back to world coordinates (voxel centers) - only for computation
        valid_voxels_compute = valid_indices_unique.float() * self.grid_size + pc_min + self.grid_size / 2

        # Calculate distances between valid voxels and anchors - preserves gradients to anchors_xyz
        dist_sq = torch.cdist(valid_voxels_compute, anchors_xyz, p=2).pow(2)
        influence_radius_sq = self.radius ** 2
        influence_scores = torch.exp(-dist_sq / (influence_radius_sq + 1e-4))
        aggregated_influence_per_voxel = torch.sum(influence_scores, dim=1)
        occupancy_probs = 1.0 - torch.exp(-aggregated_influence_per_voxel)

        # Create full occupancy grid using scatter operation
        linear_indices = (valid_indices_unique[:, 0] * g_w * g_d +
                          valid_indices_unique[:, 1] * g_d +
                          valid_indices_unique[:, 2])
        occupancy_flat = torch.zeros(g_h * g_w * g_d, device=device, dtype=torch.float32)
        occupancy_flat.scatter_(0, linear_indices, occupancy_probs)

        return occupancy_flat.reshape(1, g_h, g_w, g_d)  # [1, 60, 60, 36]


