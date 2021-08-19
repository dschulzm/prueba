import os

import cv2
import tensorflow as tf
import numpy as np
import random
from datetime import datetime
from tensorflow.keras.applications import mobilenet_v2, resnet50
# import cv2
from tensorflow.keras.models import Model, load_model
import time
from sklearn import metrics
# import itertools
from numpy.random import default_rng
from util import dataset_to_dict, plot_tsne
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import json
import socket
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

AUTOTUNE = tf.data.experimental.AUTOTUNE

# Force to use CPU
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


def preprocess_images(images, labels, preprocesor):
    return preprocesor(images), labels


def load_image(filename, input_shape):
    label = tf.strings.to_number(filename[1], out_type=tf.dtypes.int64)
    filename = filename[0]

    image = tf.io.read_file(filename)
    image = tf.image.decode_image(image, expand_animations=False)
    image = tf.image.resize(image, input_shape[0:2])
    # image = tf.cast(image, tf.uint8)
    return image, label


def parse_image(filename, input_shape, preprocessor):
    label = tf.strings.to_number(filename[1], out_type=tf.dtypes.int64)
    filename = filename[0]

    image = tf.io.read_file(filename)
    image = tf.image.decode_image(image, expand_animations=False)
    image = tf.image.resize(image, input_shape[0:2])
    # image = tf.cast(image, tf.uint8)
    image = preprocessor(image)
    return image, label


def predict_from_templates(model, images, templates):

    templates_feat = model.predict(templates)
    test_feat = model.predict(images)

    predictions = list()
    for n, feat in enumerate(test_feat):
        dist = np.linalg.norm(feat - templates_feat, axis=1)
        y_pred = dist.mean()
        predictions.append(y_pred)

    return np.asarray(predictions)
    # return np.asarray(np.max(0, 1-predictions)) # Scores between 0 and 1


def predict_from_few_shots(model, images, templates, template_labels):
    # model.trainable = False

    templates_feat = model.predict(templates)
    test_feat = model.predict(images)

    predictions = list()
    for n, feat in enumerate(test_feat):
        dist = np.linalg.norm(feat - templates_feat, axis=1)
        y_pred = template_labels[np.argmin(dist)]
        predictions.append(y_pred)

    return np.asarray(predictions)


def eval_siamese_network(model_path, templates_path, test_path, n_templates, display_labels, results_path):
    # Set seeds for repeatability
    seed = 0
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

    # Try to load json with parameters
    backbone = None
    input_shape = (224, 224, 3)
    params_path = os.path.join(model_path, 'params.json')
    if os.path.exists(params_path):
        with open(params_path) as f:
            params = json.load(f)
        backbone = params['backbone']
        input_shape = params['input_shape']

    if backbone == 'resnet50':
        preprocessor = resnet50.preprocess_input
    elif backbone == 'mobilenet_v2':
        preprocessor = mobilenet_v2.preprocess_input
    else:
        preprocessor = mobilenet_v2.preprocess_input

    # Parameters
    batch_size = 128

    # Save results to disk
    # results_path = datetime.today().strftime('%Y%m%d_%H%M%S')
    # results_path = os.path.join(model_path, results_path + '_eval')
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Templates dataset
    dataset_templates = dataset_to_dict(templates_path, n_samples_per_class=n_templates)
    random.shuffle(dataset_templates)
    dataset_templates_digital = [x for x in dataset_templates if x['label'] == 0][0:n_templates]
    dataset_templates_digital = tf.data.Dataset.from_tensor_slices(
        [[x['id'], str(x['label'])] for x in dataset_templates_digital]).map(
        lambda x: parse_image(x, input_shape, preprocessor)).batch(batch_size)
    dataset_templates = tf.data.Dataset.from_tensor_slices([[x['id'], str(x['label'])] for x in dataset_templates]).map(
        lambda x: parse_image(x, input_shape, preprocessor)).batch(batch_size)

    # Test dataset
    dataset_test = dataset_to_dict(test_path)
    dataset_test = tf.data.Dataset.from_tensor_slices([[x['id'], str(x['label'])] for x in dataset_test]).map(
        lambda x: parse_image(x, input_shape, preprocessor)).batch(batch_size)

    model = load_model(model_path, compile=False)

    test_labels = np.concatenate([y for x, y in dataset_test], axis=0)
    test_labels_binary = (test_labels == 0).astype(np.int).tolist()
    template_labels = np.concatenate([y for x, y in dataset_templates], axis=0)

    results = dict()
    few_shot_predictions = predict_from_few_shots(model, dataset_test, dataset_templates, template_labels)
    fpr, tpr, thresholds = metrics.roc_curve(test_labels_binary, (np.asarray(few_shot_predictions) == 0).astype(np.int))
    best_ndx = np.argmax(tpr - fpr)
    results['fpr_few_shots'] = fpr[best_ndx]
    results['tpr_few_shots'] = tpr[best_ndx]

    distances = predict_from_templates(model, dataset_test, dataset_templates_digital)
    fpr, tpr, thresholds = metrics.roc_curve(test_labels_binary, [(1 - x) for x in distances])
    auc = metrics.roc_auc_score(test_labels_binary, [(1 - x) for x in distances])
    best_ndx = np.argmax(tpr - fpr)

    results['fpr'] = fpr[best_ndx]
    results['tpr'] = tpr[best_ndx]
    results['thresholds'] = thresholds[best_ndx]
    results['auc'] = auc

    test_feat = model.predict(dataset_test)
    plot_tsne(test_feat, test_labels, labels=display_labels, path_save=os.path.join(results_path, 'tsne_2d.jpg'))

    roc_display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    plt.grid()
    plt.title('ROC curve using binary classification')
    plt.savefig(os.path.join(results_path, 'roc_curve.jpg'), dpi=150)

    cm = metrics.confusion_matrix(test_labels, few_shot_predictions)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot()
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(results_path, 'conf_matrix.jpg'), dpi=150)

    tmp_dict = dict()
    tmp_dict['model_path'] = model_path
    tmp_dict['templates_path'] = templates_path
    tmp_dict['test_path'] = test_path
    tmp_dict['input_shape'] = input_shape
    tmp_dict['n_templates'] = n_templates
    with open(os.path.join(results_path, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump({**tmp_dict, **results}, f, ensure_ascii=False, indent=2)

    tf.keras.backend.clear_session()
    return results


def augment_images(images, seq_name='seq_color'):
    if seq_name == 'seq_color':
        ops = [[tf.image.random_contrast, dict(lower=0.3, upper=1.25)],
               [tf.image.random_hue, dict(max_delta=0.15)],
               [tf.image.random_saturation, dict(lower=0.25, upper=3)],
               [tf.image.adjust_gamma, dict(gamma=(2 * np.random.random() + 1) ** [-1, 1][np.random.randint(2)])]]
    elif seq_name == 'seq_test':
        ops = []

    # data = next(dataset_it)[0]
    # print(data.shape)
    ndx_op = np.random.randint(0, len(ops))
    images_aug = ops[ndx_op][0](**{**dict(image=images), **ops[ndx_op][1]})
    images_aug = tf.clip_by_value(images_aug, 0, 255)
    return images_aug


def train_siamese_network(**params):
    input_shape = params['input_shape']
    batch_size = params['batch_size']
    weights = params['weights']
    epochs = params['epochs']

    exp_dir = datetime.today().strftime('%Y%m%d_%H%M%S') + '_' + socket.gethostname()
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    # log = Logger(os.path.join(exp_dir, 'output.log'))

    # Read backbone_model network
    if params['backbone'] == 'mobilenetv2':
        alpha = params['backbone_params']['alpha']
        backbone_model = mobilenet_v2.MobileNetV2(include_top=True, input_shape=input_shape, alpha=alpha, weights=weights)
        backbone_model = Model(inputs=backbone_model.input, outputs=backbone_model.layers[-2].output)
        preprocessor = mobilenet_v2.preprocess_input
    elif params['backbone'] == 'resnet50':
        backbone_model = resnet50.ResNet50(include_top=True, input_shape=input_shape, weights=weights)
        backbone_model = Model(inputs=backbone_model.input, outputs=backbone_model.layers[-2].output)
        preprocessor = resnet50.preprocess_input
    # elif params['backbone'] == 'mobilenetv2_small':
    #     # preprocessor = mobilenet_v3.preprocess_input
    # elif params['backbone'] == 'mobilenetv2_large':
    #     # preprocessor = mobilenet_v3.preprocess_input

    output = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name='l2_norm')(backbone_model.output)
    model = Model(inputs=backbone_model.input, outputs=output)

    if 'unfreeze_from' in params.keys():
        if params['unfreeze_from'] is not None:
            for layer in model.layers:
                if layer.name == params['unfreeze_from']:
                    break
                else:
                    layer.trainable = False
    # for layer in model.layers:
    #     print(layer.name, layer.trainable)

    if 'path_train' in params.keys():
        dataset_train = dataset_to_dict(params['path_train'])
        # dataset_train = dataset_to_dict(params['path_train'], n_samples_per_class=128)
        n_samples_train = len(dataset_train)
        params['n_samples_train'] = n_samples_train
        dataset_train = tf.data.Dataset.from_tensor_slices([[x['id'], str(x['label'])] for x in dataset_train])
        dataset_train = dataset_train.shuffle(n_samples_train).map(lambda x: load_image(x, input_shape),
                                      num_parallel_calls=AUTOTUNE).batch(batch_size)
        if 'data_augmentation' in params.keys():
            if params['data_augmentation'] is not None:
                dataset_train = dataset_train.map(lambda x, y: (augment_images(x, params['data_augmentation']), y),
                                                  num_parallel_calls=AUTOTUNE)
        dataset_train = dataset_train.map(lambda x, y: (preprocessor(x), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    if 'path_val' in params.keys():
        dataset_val = dataset_to_dict(params['path_val'])
        n_samples_val = len(dataset_val)
        params['n_samples_val'] = n_samples_val
        dataset_val = tf.data.Dataset.from_tensor_slices([[x['id'], str(x['label'])] for x in dataset_val])
        dataset_val = dataset_val.map(lambda x: parse_image(x, input_shape, preprocessor),
                                      num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)

    learning_rate = params['optimizer_params']['learning_rate']
    if params['optimizer'] == 'adam':
        optimizer = tf.keras.optimizers.Adam(lr=learning_rate)

    if params['loss'] == 'contrastive_loss':
        loss_fn = tfa.losses.ContrastiveLoss()
    elif params['loss'] == 'triplet_semi_hard_loss':
        loss_fn = tfa.losses.TripletSemiHardLoss()
    if params['loss'] == 'triplet_hard_loss':
        loss_fn = tfa.losses.TripletHardLoss()

    model.compile(loss=loss_fn, optimizer=optimizer)
    model.summary()

    # Path to models (Best and latest)
    best_model_path = os.path.join(exp_dir, 'best_model')
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
    latest_model_path = os.path.join(exp_dir, 'latest_model')
    if not os.path.exists(latest_model_path):
        os.makedirs(latest_model_path)

    # Callbacks
    callbacks = list()
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=latest_model_path))
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=best_model_path, save_best_only=True, monitor='val_loss'))
    # callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=os.path.join(exp_dir, 'logs')))

    # Save params to disk
    with open(os.path.join(best_model_path, 'params.json'), 'w', encoding='utf-8') as f:
        json.dump(params, f, ensure_ascii=False, indent=2)
    with open(os.path.join(latest_model_path, 'params.json'), 'w', encoding='utf-8') as f:
        json.dump(params, f, ensure_ascii=False, indent=2)

    # Train the network
    history = model.fit(
        dataset_train,
        validation_data=dataset_val,
        epochs=epochs,
        callbacks=callbacks)

    # Save training plot
    for key in sorted(history.history.keys()):
        plt.plot(range(1, len(history.history[key])+1), history.history[key], label=key)
    plt.grid()
    plt.legend()
    plt.xlabel('epochs')
    plt.title('Training evolution')
    plt.savefig(os.path.join(best_model_path, 'evolution.jpg'), dpi=150)
    plt.savefig(os.path.join(latest_model_path, 'evolution.jpg'), dpi=150)

    # Eval model
    if 'path_test' in params.keys():
        path_templates = params['path_templates'] if 'path_templates' in params.keys() else params['path_train']
        n_templates = params['n_templates'] if 'n_templates' in params.keys() else 4
        display_labels = params['display_labels']  if 'display_labels' in params.keys() else ['d', 'b', 'p', 's']

        results_path = os.path.join(best_model_path, 'results')
        eval_siamese_network(best_model_path, path_templates, params['path_test'], n_templates, display_labels,
                             results_path)
        tf.keras.backend.clear_session()

        results_path = os.path.join(latest_model_path, 'results')
        eval_siamese_network(latest_model_path, path_templates, params['path_test'], n_templates, display_labels,
                             results_path)

        tf.keras.backend.clear_session()
    # log.close()
    plt.clf()


def main():
    # Set seeds for repeatability
    seed = 0
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

    # Training parametes
    params = dict()
    params['path_train'] = list()
    params['path_train'].append('/home/dschulz/TOC/fakeid/datasets/segmented/mex1/train_300_negs.txt')
    params['path_train'].append('/home/dschulz/TOC/fakeid/datasets/segmented/arg2/train.txt')
    params['path_train'].append('/home/dschulz/TOC/fakeid/datasets/segmented/chl2/train.txt')
    params['path_train'].append('/home/dschulz/TOC/fakeid/datasets/segmented/chl1/train.txt')
    params['path_val'] = '/home/dschulz/TOC/fakeid/datasets/segmented/mex1/validation.txt'
    # params['loss'] = 'contrastive_loss'
    params['loss'] = 'triplet_semi_hard_loss'
    # params['loss'] = 'triplet_hard_loss'
    params['backbone'] = 'mobilenetv2'
    params['backbone_params'] = {'alpha': 1.0}
    params['unfreeze_from'] = 'block_2_expand'
    # params['backbone'] = 'resnet50'
    # params['unfreeze_from'] = 'conv5_block1_1_conv'
    params['input_shape'] = (224, 224, 3)
    params['batch_size'] = 128
    params['epochs'] = 100
    params['weights'] = 'imagenet' # None, 'imagenet'
    params['optimizer'] = 'adam'
    # params['optimizer_params'] = {'learning_rate': 1e-5}
    params['optimizer_params'] = {'learning_rate': 1e-5}
    params['data_augmentation'] = 'seq_color'
    params['description'] = 'MobileNetV2 unfreeze from block_2_expand: CHL1 + CHL2 + AG2 + MEX1 (300 negs)'
    # Eval parameters
    params['path_test'] = '/home/dschulz/TOC/fakeid/datasets/segmented/mex1/test.txt'
    params['path_templates'] = '/home/dschulz/TOC/fakeid/datasets/segmented/mex1/train.txt'
    params['n_templates'] = 4
    params['display_labels'] = ['d', 'b', 'p', 's']
    train_siamese_network(**params)
    exit()

    params['path_train'] = list()
    params['path_train'].append('/home/dschulz/TOC/fakeid/datasets/segmented/mex1/train_150_negs.txt')
    params['path_train'].append('/home/dschulz/TOC/fakeid/datasets/segmented/arg2/train.txt')
    params['path_train'].append('/home/dschulz/TOC/fakeid/datasets/segmented/chl2/train.txt')
    params['path_train'].append('/home/dschulz/TOC/fakeid/datasets/segmented/chl1/train.txt')
    params['description'] = 'MobileNetV2 unfreeze from block_14_expand: CHL1 + CHL2 + AG2 + MEX1 (150 negs)'
    train_siamese_network(**params)

    params['path_train'] = list()
    params['path_train'].append('/home/dschulz/TOC/fakeid/datasets/segmented/mex1/train_300_negs.txt')
    params['path_train'].append('/home/dschulz/TOC/fakeid/datasets/segmented/arg2/train.txt')
    params['path_train'].append('/home/dschulz/TOC/fakeid/datasets/segmented/chl2/train.txt')
    params['path_train'].append('/home/dschulz/TOC/fakeid/datasets/segmented/chl1/train.txt')
    params['unfreeze_from'] = 'block_13_expand'
    params['description'] = 'MobileNetV2 unfreeze from block_13_expand: CHL1 + CHL2 + AG2 + MEX1 (300 negs)'
    train_siamese_network(**params)

    params['path_train'] = list()
    params['path_train'].append('/home/dschulz/TOC/fakeid/datasets/segmented/mex1/train_150_negs.txt')
    params['path_train'].append('/home/dschulz/TOC/fakeid/datasets/segmented/arg2/train.txt')
    params['path_train'].append('/home/dschulz/TOC/fakeid/datasets/segmented/chl2/train.txt')
    params['path_train'].append('/home/dschulz/TOC/fakeid/datasets/segmented/chl1/train.txt')
    params['description'] = 'MobileNetV2 unfreeze from block_13_expand: CHL1 + CHL2 + AG2 + MEX1 (150 negs)'
    train_siamese_network(**params)

    params['path_train'] = list()
    params['path_train'].append('/home/dschulz/TOC/fakeid/datasets/segmented/mex1/train_300_negs.txt')
    params['path_train'].append('/home/dschulz/TOC/fakeid/datasets/segmented/arg2/train.txt')
    params['path_train'].append('/home/dschulz/TOC/fakeid/datasets/segmented/chl2/train.txt')
    params['path_train'].append('/home/dschulz/TOC/fakeid/datasets/segmented/chl1/train.txt')
    params['unfreeze_from'] = 'block_12_expand'
    params['description'] = 'MobileNetV2 unfreeze from block_12_expand: CHL1 + CHL2 + AG2 + MEX1 (300 negs)'
    train_siamese_network(**params)

    params['path_train'] = list()
    params['path_train'].append('/home/dschulz/TOC/fakeid/datasets/segmented/mex1/train_150_negs.txt')
    params['path_train'].append('/home/dschulz/TOC/fakeid/datasets/segmented/arg2/train.txt')
    params['path_train'].append('/home/dschulz/TOC/fakeid/datasets/segmented/chl2/train.txt')
    params['path_train'].append('/home/dschulz/TOC/fakeid/datasets/segmented/chl1/train.txt')
    params['description'] = 'MobileNetV2 unfreeze from block_12_expand: CHL1 + CHL2 + AG2 + MEX1 (150 negs)'
    train_siamese_network(**params)

    params['path_train'] = list()
    params['path_train'].append('/home/dschulz/TOC/fakeid/datasets/segmented/mex1/train_300_negs.txt')
    params['path_train'].append('/home/dschulz/TOC/fakeid/datasets/segmented/arg2/train.txt')
    params['path_train'].append('/home/dschulz/TOC/fakeid/datasets/segmented/chl2/train.txt')
    params['path_train'].append('/home/dschulz/TOC/fakeid/datasets/segmented/chl1/train.txt')
    params['unfreeze_from'] = 'block_11_expand'
    params['description'] = 'MobileNetV2 unfreeze from block_11_expand: CHL1 + CHL2 + AG2 + MEX1 (300 negs)'
    train_siamese_network(**params)

    params['path_train'] = list()
    params['path_train'].append('/home/dschulz/TOC/fakeid/datasets/segmented/mex1/train_150_negs.txt')
    params['path_train'].append('/home/dschulz/TOC/fakeid/datasets/segmented/arg2/train.txt')
    params['path_train'].append('/home/dschulz/TOC/fakeid/datasets/segmented/chl2/train.txt')
    params['path_train'].append('/home/dschulz/TOC/fakeid/datasets/segmented/chl1/train.txt')
    params['description'] = 'MobileNetV2 unfreeze from block_11_expand: CHL1 + CHL2 + AG2 + MEX1 (150 negs)'
    train_siamese_network(**params)


if __name__ == '__main__':
    main()