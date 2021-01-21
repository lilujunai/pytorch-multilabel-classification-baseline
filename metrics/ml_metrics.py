import numpy as np
from easydict import EasyDict
from sklearn.metrics import average_precision_score


def calc_average_precision(gt_label, probs):
    ndata, nattr = gt_label.shape

    ap_list = []
    for i in range(nattr):
        y_true = gt_label[:, i]
        y_score = probs[:, i]
        ap_list.append(average_precision_score(y_true, y_score))
    ap = np.array(ap_list)
    mAp = ap.mean()
    return mAp, ap


def get_map_metrics(gt_label, probs):
    mAP, ap = calc_average_precision(gt_label, probs)

    return mAP, ap


# same as calc_average_precision
def get_mAp(gt_label: np.ndarray, probs: np.ndarray):
    ndata, nattr = gt_label.shape
    rg = np.arange(1, ndata + 1).astype(float)
    ap_list = []
    for k in range(nattr):
        # sort scores
        scores = probs[:, k]
        targets = gt_label[:, k]
        sorted_idx = np.argsort(scores)[::-1]  # Descending
        truth = targets[sorted_idx]

        tp = np.cumsum(truth).astype(float)
        # compute precision curve
        precision = tp / rg

        # compute average precision
        ap_list.append(precision[truth == 1].sum() / max(truth.sum(), 1))

    ap = np.array(ap_list)
    mAp = ap.mean()
    return mAp, ap


def prob2metric(gt_label: np.ndarray, probs: np.ndarray, th):
    eps = 1e-6
    ndata, nattr = gt_label.shape
    # ------------------ ap ---------------
    # rg = np.arange(1, ndata + 1).astype(float)
    # rg = np.expand_dims(rg, axis=1)
    # rg = np.repeat(rg, nattr, axis=1)
    #
    #
    # sorted_idx = np.argsort(probs, axis=0)[::-1]  # some of the strides of a given numpy array are negative.
    # truth = torch.gather(torch.from_numpy(gt_label), 0, torch.from_numpy(sorted_idx.copy())).numpy()
    # tp = np.cumsum(truth, axis=0).astype(float)
    # precision = tp / rg
    # mask = truth == 1
    # ap = (precision * mask).sum(0) / np.maximum(truth.sum(0), 1)
    # mAp = ap.mean()

    # ------------------ macro, micro ---------------
    # gt_label[gt_label == -1] = 0
    pred_label = probs > th
    gt_pos = gt_label.sum(0)
    pred_pos = pred_label.sum(0)
    tp = (gt_label * pred_label).sum(0)

    OP = tp.sum() / pred_pos.sum()
    OR = tp.sum() / gt_pos.sum()
    OF1 = (2 * OP * OR) / (OP + OR)

    pred_pos[pred_pos == 0] = 1

    CP_all = tp / pred_pos
    CR_all = tp / gt_pos

    CP_all_t = tp / pred_pos
    CP_all_t[CP_all_t == 0] = 1
    CR_all_t = tp / gt_pos
    CR_all_t[CR_all_t == 0] = 1
    CF1_all = (2 * CP_all * CR_all) / (CP_all_t + CR_all_t)

    CF1_mean = CF1_all.mean()

    CP = np.mean(tp / pred_pos)
    CR = np.mean(tp / gt_pos)
    CF1 = (2 * CP * CR) / (CP + CR)

    gt_neg = ndata - gt_pos
    tn = ((1 - gt_label) * (1 - pred_label)).sum(0)

    label_pos_recall = 1.0 * tp / (gt_pos + eps)  # true positive
    label_neg_recall = 1.0 * tn / (gt_neg + eps)  # true negative
    # mean accuracy
    label_ma = (label_pos_recall + label_neg_recall) / 2

    ma = label_ma.mean()

    return OP, OR, OF1, CP, CR, CF1, ma, CP_all, CR_all, CF1_all, CF1_mean


def get_multilabel_metrics(gt_label, prob_pred, th=0.5):
    result = EasyDict()

    mAP, ap = calc_average_precision(gt_label, prob_pred)
    op, orecall, of1, cp, cr, cf1, ma, cp_all, cr_all, cf1_all, CF1_mean = prob2metric(gt_label, prob_pred, th)
    result.map = mAP * 100.

    # to json serializable
    result.CP_all = list(cp_all.astype(np.float64))
    result.CR_all = list(cr_all.astype(np.float64))
    result.CF1_all = list(cf1_all.astype(np.float64))
    result.CF1_mean = CF1_mean

    # simplified way
    # mAP, ap = calc_average_precision(gt_label, probs)
    # pred_label = probs > 0.5
    # CP, CR, _, _ = precision_recall_fscore_support(gt_label, pred_label, average='macro')
    # CF1 = 2 * CP * CR / (CP + CR)
    # OP, OR, OF1, _ = precision_recall_fscore_support(gt_label, pred_label, average='micro')

    result.OP = op * 100.
    result.OR = orecall * 100.
    result.OF1 = of1 * 100.
    result.CP = cp * 100.
    result.CR = cr * 100.
    result.CF1 = cf1 * 100.

    # result.ma = ma * 100.

    return result


if __name__ == '__main__':
    demo_probs = np.random.randn(5, 5)
    demo_probs = (demo_probs - demo_probs.min()) / (demo_probs - demo_probs.min()).max()
    demo_gt = np.random.randint(0, 2, (5, 5))

    get_multilabel_metrics(demo_gt, demo_probs)
    # print(get_mAp(demo_gt, demo_probs))

    # print(get_mAp_macro_micro(demo_probs, demo_gt))
