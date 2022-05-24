from fakeid_classification import FakeID2
from fakeid_segmentation import FakeIDSegmentation
from fakeid_remote import FakeIDRemote
import sys
import cv2
import os
import numpy as np
import random
import time
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Force to use CPU
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = ''


sys.path.append('/home/dschulz/TOC/aikit/')
from aikit.graphics.biometric_performance import performance_evaluation
from aikit.graphics.confusion_matrix import (
    plot_confusion_matrix,
    plot_system_confusion_matrix
)
from aikit.graphics.det_plot import DETPlot
from aikit.metadata import __version__ as aikit_ver
from aikit.metrics.det_curve import det_curve_pais, eer_pais
from aikit.metrics.iso_30107_3 import (
    acer,
    apcer_max,
    apcer_pais,
    bpcer,
    bpcer_ap,
    riapar
)
from aikit.metrics.scores import (
    max_error_pais_scores,
    pad_scores,
    split_attack_scores,
    split_scores
)


def dataset_to_dict(input_data, shuffle_list=True, n_samples_per_class=None, remap_labels=None):
    file_ext = ['.jpg', '.JPG', '.png', '.PNG', '.jp2', '.JP2']
    dataset = list()

    if isinstance(input_data, list):
        for file in input_data:
            f = open(file, 'rt')
            for line in f:
                file_path, label = line.strip('\n').split(' ')
                sample_dict = dict()
                sample_dict['id'] = file_path
                sample_dict['label'] = int(label)
                dataset.append(sample_dict)
            f.close()
    elif os.path.isfile(input_data):
        f = open(input_data, 'rt')
        for line in f:
            file_path, label = line.strip('\n').split(' ')
            sample_dict = dict()
            sample_dict['id'] = file_path
            sample_dict['label'] = int(label)
            dataset.append(sample_dict)
        f.close()
    elif os.path.isdir(input_data):
        file_paths = list()
        for path, subdirs, files in os.walk(input_data):
            for name in files:
                if os.path.splitext(name)[1] in file_ext:
                    file_paths.append(os.path.join(path, name))
        base_dirs = list()
        for file_path in file_paths:
            base_dir = os.path.basename(os.path.split(file_path)[0])
            base_dirs.append(base_dir)
        base_dirs = sorted(list(set(base_dirs)))
        for file_path in file_paths:
            label = base_dirs.index(os.path.basename(os.path.split(file_path)[0]))
            sample_dict = dict()
            sample_dict['id'] = file_path
            sample_dict['label'] = int(label)
            dataset.append(sample_dict)

    if remap_labels is not None:
        for data in dataset:
            data['label'] = remap_labels[data['label']]

    if n_samples_per_class is not None:
        dataset_new = list()
        labels_set = sorted(list(set([x['label'] for x in dataset])))
        for lbl in labels_set:
            dataset_lbl = [x for x in dataset if x['label'] == lbl]
            # print(lbl, len(dataset_lbl))
            dataset_lbl = dataset_lbl[:n_samples_per_class]
            # print(lbl, len(dataset_lbl))
            dataset_new += dataset_lbl
        dataset = dataset_new

    if shuffle_list is True:
        random.shuffle(dataset)

    print(input_data)
    labels_set = sorted(list(set([x['label'] for x in dataset])))
    for lbl in labels_set:
        print('label: %d n_samples: %d' % (lbl, len([x for x in dataset if x['label'] == lbl])))

    return dataset


def eval(labels, scores, threshold, results_path):
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Evaluation
    labels = np.array(labels)
    scores = np.array(scores)

    attack_scores, bonafide_scores, attack_true, bonafide_true = split_scores(labels, scores, bonafide_label=0)
    pais_attack_scores = split_attack_scores(attack_true, attack_scores)

    det_pais = det_curve_pais(attack_true, attack_scores, bonafide_scores)
    eer_pais_ = eer_pais(det_pais, percentage=True)

    max_eer_pais = max(eer_pais_, key=eer_pais_.get)
    max_attack_scores, max_attack_pais = max_error_pais_scores(attack_true, attack_scores, threshold=threshold)

    acer_ = acer(attack_true, attack_scores, bonafide_scores, threshold=threshold)
    apcer_ = apcer_pais(attack_true, attack_scores, threshold=threshold, percentage=True)
    bpcer_ = bpcer(bonafide_scores, threshold=threshold)

    bpcer_ap_all = {key: None for key in [10, 20, 50, 100, 200, 500, 1000, 10000]}
    for val in bpcer_ap_all.keys():
        bpcer_ap_pais = dict()
        for pais in eer_pais_.keys():
            bpcer_ap_, bpcer_ap_thres = bpcer_ap(det_pais[pais][0], det_pais[pais][1], det_pais[pais][2], val, percentage=True)
            bpcer_ap_pais[pais] = (bpcer_ap_, bpcer_ap_thres)
        max_bpcer_ap = max(bpcer_ap_pais, key=bpcer_ap_pais.get)
        bpcer_ap_all[val] = bpcer_ap_pais[max_bpcer_ap]
    riapar_ = riapar(max_attack_scores, bonafide_scores, attack_threshold=threshold, bonafide_threshold=threshold)

    classes = np.array(["digital", "border", "printed", "screen", "plastic", "synthetic"])
    # classes = np.array(display_labels)
    bf_label = 0
    f = open(os.path.join(results_path, 'report.txt'), 'wt')
    f.write(
        f"        Bona Fide label: {bf_label}: {classes[bf_label]}\n"
        f"            Threshold t: {threshold}\n"
        "--------------------------------------------\n"
        f"           Max EER PAIS: {max_eer_pais}: {classes[max_eer_pais]}\n"
        f"                 EER[{max_eer_pais}]: {eer_pais_[max_eer_pais][0]}%\n"
        f"       EER threshold[{max_eer_pais}]: {eer_pais_[max_eer_pais][1]}\n"
        "--------------------------------------------\n"
        f"      Max APCER PAIS(t): {max_attack_pais}: {classes[max_attack_pais]}\n"
        f"                ACER(t): {acer_ * 100}%\n"
        f"               APCER(t): {apcer_}%\n"
        f"               BPCER(t): {bpcer_ * 100}%\n"
        f"              RIAPAR(t): {riapar_ * 100}%\n"
        f"                 EER[{max_attack_pais}]: {eer_pais_[max_attack_pais][0]}%\n"
        f"       EER threshold[{max_attack_pais}]: {eer_pais_[max_attack_pais][1]}\n"
        "--------------------------------------------\n"
    )
    for val in bpcer_ap_all.keys():
        f.write(f"   BPCER{val}(APCER={100/val}%): {bpcer_ap_all[val][0]}%, {bpcer_ap_all[val][1]}\n")

    f.close()

    # Create results plots
    det = DETPlot(title="DET Curve")
    det.set_system_pais(attack_true, attack_scores, bonafide_scores, pais_names=classes, label="Fake-ID 2.0")
    det_ = det.plot()
    det_.savefig(os.path.join(results_path, 'det.jpg'))

    cmap = sns.cubehelix_palette(start=0.5, rot=-0.75, gamma=1.2, hue=1.25, dark=0.15, reverse=True, as_cmap=True)
    cm2 = plot_system_confusion_matrix(
        attack_scores,
        bonafide_scores,
        threshold=threshold,
        fontsize=30,
        cmap=cmap
    )
    cm2.savefig(os.path.join(results_path, 'conf_matrix_binary.jpg'))

    padded_attack_scores, padded_bonafide_scores = pad_scores(attack_scores, bonafide_scores)
    df = pd.DataFrame({'Bona fide scores': padded_bonafide_scores, 'Attack presentation scores': padded_attack_scores})
    kde = performance_evaluation(df, threshold=threshold, figsize=(10, 10))
    kde.savefig(os.path.join(results_path, 'kde.jpg'))
    kde2 = performance_evaluation(df, threshold=threshold, log_scale=True, figsize=(10, 10))
    kde2.savefig(os.path.join(results_path, 'kde2.jpg'))


def main():
    REMOTE_EVAL = False
    fakeid_path = '../releases/MEX-2021-12/models/fakeid/'
    url_remote = 'http://54.145.73.43:5000/api/v3/document/fake/detector'
    segmentator_path = '../releases/MEX-2021-12/models/segmentation/MobileUNet_TF22_ALL_V181121.h5'
    dataset_path = ['/home/dschulz/TOC/fakeid/datasets/20210906-ARG-NOSEG/arg1/test.txt',
                    '/home/dschulz/TOC/fakeid/datasets/20210906-ARG-NOSEG/arg2/test.txt',
                    '/home/dschulz/TOC/fakeid/datasets/chl_2021-10-08/test.txt',
                    '/home/dschulz/TOC/fakeid/datasets/mex/test.txt',
                    '/home/dschulz/TOC/fakeid/datasets/segmented/uru/test.txt']

    # Instantiate models
    fakeid_remote = FakeIDRemote(url_remote)
    fakeid_model = FakeID2(fakeid_path)
    segmentation_model = FakeIDSegmentation(segmentator_path)

    # Create folder for saving results to disk
    results_path = os.path.join(fakeid_path, 'prod_results')
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    dataset = dataset_to_dict(dataset_path)
    if REMOTE_EVAL:
        warnings.warn('!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Performing remote evaluation !!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    labels = list()
    predictions = list()
    scores = list()
    f = open(os.path.join(results_path, 'scores.txt'), 'wt')
    t0 = time.time()
    for data in tqdm(dataset):
        if REMOTE_EVAL:
            pred, score = fakeid_remote.classify(data['id'])
        else:
            image = cv2.cvtColor(cv2.imread(data['id']), cv2.COLOR_BGR2RGB)
            image = segmentation_model.forward(image)
            image = fakeid_model.process_image(image)
            pred, score = fakeid_model.classify(image)

        labels.append(data['label'])
        predictions.append(pred)
        scores.append(score)
        f.write('%f %d\n' % (score, data['label']))

    t1 = time.time()
    f.close()
    print('Total time:', t1 - t0, '[s]')
    print('Avg time per image:', 1000*(t1 - t0)/len(dataset), '[ms]')

    eval(labels, scores, fakeid_model.threshold, results_path)


if __name__ == '__main__':
    main()
