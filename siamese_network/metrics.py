from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d


def det_curve(y_true, y_score, pos_label=0):
    y_score_bf = y_score[y_true == pos_label]
    y_true_atk = y_true[y_true != pos_label]
    y_score_atk = y_score[y_true != pos_label]

    out = list()
    labels_atk = sorted(list(set(y_true_atk)))
    for lbl_atk in labels_atk:
        y_score_eval = np.concatenate((y_score_bf, y_score_atk[y_true_atk == lbl_atk]))
        y_true_eval = [1] * len(y_score_bf) + [0] * len(y_score_atk[y_true_atk == lbl_atk])
        fpr, tpr, thresholds = metrics.roc_curve(y_true_eval, y_score_eval, drop_intermediate=False)
        out_atk = dict()
        out_atk['lbl_atk'] = lbl_atk
        # out_atk['fpr'] = fpr[1:]
        # out_atk['fnr'] = 1 - tpr[1:]
        # out_atk['thresholds'] = thresholds[1:]
        out_atk['fpr'] = fpr
        out_atk['fnr'] = 1 - tpr
        out_atk['thresholds'] = thresholds
        out.append(out_atk)

    return out


def apcer(fpr, thresholds, threshold_eval=0.5):
    f_apcer = interp1d(thresholds, fpr)
    return float(f_apcer(threshold_eval))


def bpcer(fnr, thresholds, threshold_eval=0.5):
    f_bpcer = interp1d(thresholds, fnr)
    return float(f_bpcer(threshold_eval))


def eer(fpr, fnr, thresholds):
    step_sampling = 0.000005
    thresh_sampling = np.arange(0, thresholds.max(), step_sampling)
    f_apcer = interp1d(thresholds, fpr)
    apcer = f_apcer(thresh_sampling)
    f_bpcer = interp1d(thresholds, fnr)
    bpcer = f_bpcer(thresh_sampling)

    eer_ndx = np.argmin(np.abs(bpcer-apcer))
    return bpcer[eer_ndx], thresh_sampling[eer_ndx]


def bpcer_ap(fpr, fnr, thresholds, ap):
    step_sampling = 0.000005
    thresh_sampling = np.arange(0, thresholds.max(), step_sampling)
    f_apcer = interp1d(thresholds, fpr)
    apcer = f_apcer(thresh_sampling)
    f_bpcer = interp1d(thresholds, fnr)
    bpcer = f_bpcer(thresh_sampling)

    ap_ndx = np.argmin(np.abs(1 / ap - apcer))
    return bpcer[ap_ndx], thresh_sampling[ap_ndx]


def det_curve_new(y_true, y_score, pos_label=0):
    y_true_bf = y_true[y_true == pos_label]
    y_score_bf = y_score[y_true == pos_label]
    y_true_atk = y_true[y_true != pos_label]
    y_score_atk = y_score[y_true != pos_label]

    thresholds = np.unique(y_score)

    plt.subplot(2, 1, 1)
    colors_plot = ['r', 'b', 'c', 'm', 'y']
    labels_atk = sorted(list(set(y_true_atk)))
    # labels_atk = np.unique(y_true_atk)
    for lbl_atk in labels_atk:
        y_score_eval = np.concatenate((y_score_bf, y_score_atk[y_true_atk == lbl_atk]))
        y_true_eval = [1] * len(y_score_bf) + [0] * len(y_score_atk[y_true_atk == lbl_atk])
        fpr, tpr, thresholds = metrics.roc_curve(y_true_eval, y_score_eval, drop_intermediate=False)

        th = 0.5
        print('bpcer thresh', th, ':', np.mean(y_score_bf < th))
        print('apcer thresh', th, ':', np.mean(y_score_atk[y_true_atk == lbl_atk] >= th))
        for fp, tp, thresh in (zip(fpr, tpr, thresholds)):
            print(fp, 1-tp, thresh)
        print('--------------------')
        plt.subplot(2, 1, 1)
        plt.plot(thresholds, fpr, str(colors_plot[labels_atk.index(lbl_atk)]) + 'o-', label='APCER PAIS ' + str(lbl_atk))

        # plt.subplot(2, 1, 2)
        # plt.plot(fpr, 1 - tpr, str(colors_plot[labels_atk.index(lbl_atk)]) + 'o-', label='PAIS ' + str(lbl_atk))

    plt.subplot(2, 1, 1)
    plt.plot(thresholds, 1-tpr, 'go-', label='BPCER')
    plt.grid()
    # plt.yscale('log')
    plt.ylim(-0.01, 0.2)
    plt.xlim(0, 1)
    plt.legend()
    plt.xlabel('threshold')
    plt.hlines(0.01, 0, 1)
    plt.text(0.01, 0.01, 'AP100')
    plt.hlines(0.05, 0, 1)
    plt.text(0.01, 0.05, 'AP20')
    plt.hlines(0.1, 0, 1)
    plt.text(0.01, 0.1, 'AP10')

    plt.subplot(2, 1, 2)
    # # plt.plot(fpr, 1 - tpr, 'o-', label='PAI ' + str(lbl_atk))
    plt.xlim(0.001, 0.41)
    plt.ylim(0.001, 0.41)
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    # plt.xlabel('')
    plt.grid()

    plt.show()

    thresh_sampling = np.arange(0, 1, 0.00001)
    f_apcer = interp1d(thresholds, fpr)
    apcer = f_apcer(thresh_sampling)
    f_bpcer = interp1d(thresholds, 1-tpr)
    bpcer = f_bpcer(thresh_sampling)

    ap_eval = [10, 20, 100]
    for ap in ap_eval:
        ndx_ap = np.argmin(np.abs(1/ap -apcer))
        # print(ap, ndx_ap, apcer[ndx_ap])
        print('BPCER' + str(ap) + ': ' + str(bpcer[ndx_ap]))

    eer_ndx = np.argmin(np.abs(bpcer-apcer))
    print('EER (APCER):', apcer[eer_ndx])
    print('EER (BPCER):', bpcer[eer_ndx])

    plt.vlines(thresh_sampling[eer_ndx], 0, 1)
    plt.plot(thresh_sampling, apcer, 'r-')
    plt.plot(thresholds, fpr, 'ro')
    plt.plot(thresh_sampling, bpcer, 'g-')
    plt.plot(thresholds, 1-tpr, 'go')
    plt.grid()
    plt.xlim(0, 1)
    plt.show()

    return 0
