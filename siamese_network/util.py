import copy

import numpy as np
import cv2
import os
import random
from tensorflow.keras.utils import to_categorical
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def plot_tsne(x, y=None, labels=None, colors=None, plot_title=None, path_save=None):
    if y is None:
        y = np.zeros((x.shape[0]))
    if labels is None:
        labels = sorted(list(np.unique(y)))
    if colors is None:
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    markers = ['o', 'v', '^', '<', '>', 's', 'd', '*']

    if plot_title is None:
        plot_title = 't-SNE 2D projection'

    x_tsne = TSNE(n_components=2).fit_transform(x)
    plt.figure(figsize=(20, 10))
    # plt.figure(figsize=(14, 7))
    for n, lbl in enumerate(sorted(np.unique(y).tolist())):
        plt.scatter(x_tsne[y == lbl, 0], x_tsne[y == lbl, 1], color=colors[n % len(colors)], marker=markers[n % len(markers)], label=labels[n])
    plt.legend()
    plt.grid()

    plt.title(plot_title)
    if path_save is not None:
        plt.savefig(path_save, dpi=150)
        plt.close()
    else:
        plt.show()
        plt.close()


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


def dataset_dict_generator(input_list, target_size, batch_size, preprocessor=None, augmentator=None, n_aug=0,
                           use_rgb=True, use_categorical_labels=True):
    n_classes = len(list(set([x['label'] for x in input_list])))
    x = list()
    y = list()
    cont = 0
    while True:
        for data_dict in input_list:
            file_path = data_dict['id']
            # print(file_path, labels[n])
            # print(cont)
            img = cv2.imread(file_path)
            img = cv2.resize(img, target_size[0:2])
            if use_rgb is True:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # cv2.imshow('img', img)
            # cv2.waitKey()
            x.append(img)
            y.append(int(data_dict['label']))
            cont += 1

            if cont % batch_size == 0:
                if preprocessor is not None:
                    x_out = preprocessor(np.asarray(x).astype('float32'))
                else:
                    x_out = np.asarray(x).astype('float32') / 255

                y_out = np.asarray(y)
                if use_categorical_labels is True:
                    y_out = to_categorical(y_out, num_classes=n_classes)
                yield x_out, y_out
                for n in range(n_aug):
                    if augmentator is not None:
                        x_out = augmentator(images=np.asarray(x))
                        if preprocessor is not None:
                            x_out = preprocessor(x_out.astype('float32'))
                        else:
                            x_out = x_out.astype('float32') / 255
                        yield x_out, y_out
                x.clear()
                y.clear()
                cont = 0


def write_dataset_dict(dataset, filename):
    f = open(filename, 'wt')

    for sample in dataset:
        line = str(sample['id']) + ' ' + str(sample['label']) + '\n'
        f.write(line)

    f.close()


def oversample_dataset(dataset):
    dataset_new = list()
    labels_set = sorted(list(set([x['label'] for x in dataset])))
    n_max_samples = max([len([x for x in dataset if x['label'] == lbl]) for lbl in labels_set])
    for lbl in labels_set:
        dataset_lbl = [x for x in dataset if x['label'] == lbl]
        dataset_tmp = copy.deepcopy(dataset_lbl)
        for i in range(int(n_max_samples/len(dataset_lbl))):
            random.shuffle(dataset_lbl)
            dataset_tmp += dataset_lbl
        dataset_new += dataset_tmp[0:n_max_samples]
    random.shuffle(dataset_new)
    return dataset_new
