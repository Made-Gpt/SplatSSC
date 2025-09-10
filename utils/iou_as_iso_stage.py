import numpy as np

class SSCMetricsStage:
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.reset()

    def hist_info(self, n_cl, pred, gt):
        assert pred.shape == gt.shape
        k = (gt >= 0) & (gt < n_cl)  # exclude 255
        labeled = np.sum(k)
        correct = np.sum((pred[k] == gt[k]))

        return (
            np.bincount(
                n_cl * gt[k].astype(int) + pred[k].astype(int), minlength=n_cl ** 2
            ).reshape(n_cl, n_cl),
            correct,
            labeled,
        )

    @staticmethod
    def compute_score(hist, correct, labeled):
        iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
        mean_IU = np.nanmean(iu)
        mean_IU_no_back = np.nanmean(iu[1:])
        freq = hist.sum(1) / hist.sum()
        freq_IU = (iu[freq > 0] * freq[freq > 0]).sum()
        mean_pixel_acc = correct / labeled if labeled != 0 else 0

        return iu, mean_IU, mean_IU_no_back, mean_pixel_acc

    def add_batch(self, y_pred, y_true, nonempty=None, nonsurface=None, fov_mask=None):
        self.count += 1
        mask = y_true != 0  # mask 'ignored' voxels (according to the label)

        if nonempty is not None:
            mask = mask & nonempty
        if nonsurface is not None:
            mask = mask & nonsurface
        if fov_mask is not None:
            mask = mask & fov_mask
        tp, fp, fn = self.get_score_completion(y_pred, y_true, mask)

        self.completion_tp += tp
        self.completion_fp += fp
        self.completion_fn += fn

        tp_sum, fp_sum, fn_sum = self.get_score_semantic_and_completion(
            y_pred, y_true, mask
        )
        self.tps += tp_sum
        self.fps += fp_sum
        self.fns += fn_sum

    def get_stats(self):
        if self.completion_tp != 0:
            precision = self.completion_tp / (self.completion_tp + self.completion_fp)
            recall = self.completion_tp / (self.completion_tp + self.completion_fn)
            iou = self.completion_tp / (
                self.completion_tp + self.completion_fp + self.completion_fn
            )
        else:
            precision, recall, iou = 0, 0, 0
        iou_ssc = self.tps / (self.tps + self.fps + self.fns + 1e-5)
        return {
            "precision": precision,
            "recall": recall,
            "iou": iou,
            "iou_ssc": iou_ssc,
            "iou_ssc_mean": np.mean(iou_ssc),  # FIXME, baseline is np.mean(iou_ssc[1:])
        }

    def reset(self):

        self.completion_tp = 0
        self.completion_fp = 0
        self.completion_fn = 0
        self.tps = np.zeros(self.n_classes)
        self.fps = np.zeros(self.n_classes)
        self.fns = np.zeros(self.n_classes)

        self.hist_ssc = np.zeros((self.n_classes, self.n_classes))
        self.labeled_ssc = 0
        self.correct_ssc = 0

        self.precision = 0
        self.recall = 0
        self.iou = 0
        self.count = 1e-8
        self.iou_ssc = np.zeros(self.n_classes, dtype=np.float32)
        self.cnt_class = np.zeros(self.n_classes, dtype=np.float32)

    def get_score_completion(self, predict, target, mask=None):
        predict = np.copy(predict)
        target = np.copy(target)

        _bs = predict.shape[0]
        # shape is (_bs, num_voxels)
        target = target.reshape(_bs, -1)
        predict = predict.reshape(_bs, -1)

        # non-empty and non-ignored are viewed as occupied（1），else empty (0)
        b_pred = np.zeros_like(predict)
        b_true = np.zeros_like(target)
        # b_pred[(predict > 0) & (predict != 255)] = 1
        # b_true[(target > 0) & (target != 255)] = 1

        # in prediction, 'empty' -- 11 and 'ignore' -- 255 should be 0, others are valid, should be 1
        b_pred[(predict != 0)] = 1
        # in label, 'empty' -- 11 and 'ignore' -- 255 should be 0, others are valid, should be 1
        b_true[(target != 0) & (target != 12)] = 1

        tp_sum, fp_sum, fn_sum = 0, 0, 0
        for idx in range(_bs):
            y_true = b_true[idx, :]
            y_pred = b_pred[idx, :]

            if mask is not None:
                mask_idx = mask[idx, :].reshape(-1)
                y_true = y_true[mask_idx == 1]
                y_pred = y_pred[mask_idx == 1]

            tp = np.sum((y_true == 1) & (y_pred == 1))  # real occupied
            fp = np.sum((y_true == 0) & (y_pred == 1))  # false occupied
            fn = np.sum((y_true == 1) & (y_pred != 1))  # false empty and ignored
            tp_sum += tp
            fp_sum += fp
            fn_sum += fn
        return tp_sum, fp_sum, fn_sum

    def get_score_semantic_and_completion(self, predict, target, mask=None):
        target = np.copy(target)
        predict = np.copy(predict)
        _bs = predict.shape[0]  # batch size
        _C = self.n_classes  # _C = 12 or 11

        # ---- ignore
        # predict[predict == 255] = 12  # set ignore '255' to empty '11'
        # target[target == 255] = 12  # set ignore '255' to empty '11'
        # ---- flatten
        target = target.reshape(_bs, -1)  # (_bs, 129600)
        predict = predict.reshape(_bs, -1)  # (_bs, 129600), 60*36*60=129600

        tp_sum = np.zeros(_C, dtype=np.int32)  # tp
        fp_sum = np.zeros(_C, dtype=np.int32)  # fp
        fn_sum = np.zeros(_C, dtype=np.int32)  # fn

        # mask = (target != 0) & (target != 12)

        for idx in range(_bs):
            y_true = target[idx, :]  # GT
            y_pred = predict[idx, :]

            # flat_target = target.flatten()
            # flat_predict = predict.flatten()
            # unique_labels, counts_label = np.unique(flat_target, return_counts=True)
            # unique_predicts, counts_predict = np.unique(flat_predict, return_counts=True)
            # label_counts = dict(zip(unique_labels, counts_label))
            # predict_counts = dict(zip(unique_predicts, counts_predict))
            # print("Label counts as a dictionary:", label_counts)
            # print("Predict counts as a dictionary:", predict_counts)

            if mask is not None:
                mask_idx = mask[idx, :].reshape(-1)
                y_true = y_true[mask_idx == 1]
                y_pred = y_pred[mask_idx == 1]

            for j in range(1, _C+1):
                tp = np.array(np.where(np.logical_and(y_true == j, y_pred == j))).size
                fp = np.array(np.where(np.logical_and(y_true != j, y_pred == j))).size
                fn = np.array(np.where(np.logical_and(y_true == j, y_pred != j))).size

                tp_sum[j-1] += tp  # FIXME, baseline: tp_sum[j] += tp
                fp_sum[j-1] += fp  # FIXME, baseline: fp_sum[j] += fp
                fn_sum[j-1] += fn  # FIXME, baseline: fn_sum[j] += fn
        return tp_sum, fp_sum, fn_sum

