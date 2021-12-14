import sys
import json
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import mobilenet_v2, resnet50, resnet
import os
import numpy as np
import random
sys.path.append('../siamese_network')
from siamese_network import load_image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
AUTOTUNE = tf.data.experimental.AUTOTUNE


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


def main():
    # Set seeds for repeatability
    seed = 0
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

    model_path = '/home/dschulz/TOC/fakeid_2.0/releases/kaka'

    if os.path.exists(os.path.join(model_path, 'params.json')):
        with open(os.path.join(model_path, 'params.json')) as f:
            params = json.load(f)

    templates_path = params['path_templates'] if 'path_templates' in params.keys() else os.path.join(
        os.path.split(params['path_val'])[0], 'train.txt')
    n_templates = params['n_templates'] if 'n_templates' in params.keys() else 4
    input_shape = params['input_shape'] if 'n_templates' in params.keys() else (224, 224, 3)

    # Read backbone_model network
    if params['backbone'] == 'mobilenetv2':
        preprocessor = mobilenet_v2.preprocess_input
    elif params['backbone'] == 'resnet50':
        preprocessor = resnet50.preprocess_input
    elif params['backbone'] == 'resnet101':
        preprocessor = resnet.preprocess_input
    elif params['backbone'] == 'resnet152':
        preprocessor = resnet.preprocess_input
    elif params['backbone'] == 'custom':
        preprocessor = mobilenet_v2.preprocess_input

    model = load_model(model_path, compile=False)
    model.summary()

    # Templates dataset
    dataset_templates = dataset_to_dict(templates_path, n_samples_per_class=n_templates)
    random.shuffle(dataset_templates)
    dataset_templates = [x for x in dataset_templates if x['label'] == 0][0:n_templates]
    dataset_templates = tf.data.Dataset.from_tensor_slices([[x['id'], str(x['label'])] for x in dataset_templates])
    dataset_templates = dataset_templates.map(lambda x: load_image(x, input_shape),
                                                              num_parallel_calls=AUTOTUNE).batch(128)
    dataset_templates = dataset_templates.map(lambda x, y: (preprocessor(x), y),
                                                              num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    feat_templates = model.predict(dataset_templates)
    print(feat_templates.shape)
    print(feat_templates.min(), feat_templates.max(), feat_templates.mean(), feat_templates.std())
    np.save(os.path.join(model_path, 'templates.npy'), feat_templates)


if __name__ == '__main__':
    main()