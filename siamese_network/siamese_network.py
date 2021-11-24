import os
import sys
import warnings
import shutil

sys.path.append('/home/dschulz/TOC/aikit/')
# from aikit.metrics import iso_30107_3, scores, det_curve

# import cv2
import tensorflow as tf
import numpy as np
import random
from datetime import datetime
from tensorflow.keras.applications import mobilenet_v2, resnet50, resnet
from tensorflow.keras.models import Model, load_model
from sklearn import metrics
from numpy.random import default_rng
from util import dataset_to_dict, plot_tsne, oversample_dataset
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import json
import socket
import metrics as metrics_dschulz
import callbacks as custom_callbacks
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DEBUG = False
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
    # image = tf.image.resize(image, input_shape[0:2], method=tf.image.ResizeMethod.AREA)
    # image = tf.cast(image, tf.uint8)
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


def predict_from_few_shots(model, images, templates, template_labels):
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
    batch_size = 128
    params_path = os.path.join(model_path, 'params.json')
    if os.path.exists(params_path):
        with open(params_path) as f:
            params = json.load(f)
        backbone = params['backbone']
        input_shape = params['input_shape']
        batch_size = params['batch_size']

    if backbone == 'mobilenet_v2':
        preprocessor = mobilenet_v2.preprocess_input
    elif backbone == 'resnet50':
        preprocessor = resnet50.preprocess_input
    elif backbone == 'resnet101':
        preprocessor = resnet.preprocess_input
    elif backbone == 'resnet152':
        preprocessor = resnet.preprocess_input
    else:
        preprocessor = mobilenet_v2.preprocess_input

    # Create folder for saving results to disk
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Templates dataset
    dataset_templates = dataset_to_dict(templates_path, n_samples_per_class=n_templates)
    random.shuffle(dataset_templates)
    dataset_templates_digital = [x for x in dataset_templates if x['label'] == 0][0:n_templates]
    dataset_templates_digital = tf.data.Dataset.from_tensor_slices([[x['id'], str(x['label'])] for x in dataset_templates_digital])
    dataset_templates_digital = dataset_templates_digital.map(lambda x: load_image(x, input_shape), num_parallel_calls=AUTOTUNE).batch(
        batch_size)
    dataset_templates_digital = dataset_templates_digital.map(lambda x, y: (preprocessor(x), y), num_parallel_calls=AUTOTUNE).prefetch(
        AUTOTUNE)

    dataset_templates = tf.data.Dataset.from_tensor_slices([[x['id'], str(x['label'])] for x in dataset_templates])
    dataset_templates = dataset_templates.map(lambda x: load_image(x, input_shape), num_parallel_calls=AUTOTUNE).batch(
        batch_size)
    dataset_templates = dataset_templates.map(lambda x, y: (preprocessor(x), y), num_parallel_calls=AUTOTUNE).prefetch(
        AUTOTUNE)

    # Test dataset
    dataset_test = dataset_to_dict(test_path)
    if DEBUG:
        random.shuffle(dataset_test)
        dataset_test = dataset_test[0:128]

    test_ids = [x['id'] for x in dataset_test]
    dataset_test = tf.data.Dataset.from_tensor_slices([[x['id'], str(x['label'])] for x in dataset_test])
    dataset_test = dataset_test.map(lambda x: load_image(x, input_shape), num_parallel_calls=AUTOTUNE).batch(batch_size)
    dataset_test = dataset_test.map(lambda x, y: (preprocessor(x), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

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

    results_detail = list()
    for n, id in enumerate(test_ids):
        results_detail_dict = dict()
        results_detail_dict['id'] = id
        results_detail_dict['label'] = str(test_labels[n])
        results_detail_dict['distance_templates'] = str(distances[n])
        results_detail_dict['few_shots_pred'] = str(few_shot_predictions[n])
        results_detail.append(results_detail_dict)
    with open(os.path.join(results_path, 'results_detail.json'), 'w', encoding='utf-8') as f:
        json.dump(results_detail, f, ensure_ascii=False, indent=2)

    results['fpr'] = fpr[best_ndx]
    results['tpr'] = tpr[best_ndx]
    results['thresholds'] = thresholds[best_ndx]
    results['auc'] = auc

    # ISO 30107-3 Metrics
    # distances = np.clip(distances, 0, 1)
    # atk_scores, bf_scores, atk_true, bf_true = scores.split_scores(test_labels, 1 - distances, bonafide_label=0)
    # apcer_pais = iso_30107_3.apcer_pais(atk_true, atk_scores)
    # results['apcer_pais'] = str(apcer_pais)
    # apcer_max = iso_30107_3.apcer_max(atk_true, atk_scores)
    # results['apcer_max'] = apcer_max
    # bpcer = iso_30107_3.bpcer(bf_scores)
    # results['bpcer'] = bpcer
    # apcer_, bpcer_, thresholds_ = det_curve.det_curve(atk_scores, bf_scores, False)
    # eer, eer_threshold = det_curve.eer(apcer_, bpcer_, thresholds_)
    # results['eer'] = eer
    # results['eer_threshold'] = eer_threshold
    #
    # bpcer_10 = iso_30107_3.bpcer_ap(bf_scores, apcer_, thresholds_, 10)
    # bpcer_20 = iso_30107_3.bpcer_ap(bf_scores, apcer_, thresholds_, 20)
    # bpcer_100 = iso_30107_3.bpcer_ap(bf_scores, apcer_, thresholds_, 100)
    # print('bpcer_10', 'bpcer_20', 'bpcer_100')
    # print(bpcer_10, bpcer_20, bpcer_100)

    distances = np.clip(distances, 0, 1)
    # metrics_dschulz.det_curve_new(test_labels, 1 - distances)
    ap_eval = [10, 20, 100]
    apcer_pais = dict()
    eer_pais = dict()
    bpcer_ap_pais = dict()
    det_values = metrics_dschulz.det_curve(test_labels, 1-distances)
    for pais in det_values:
        lbl_atk = pais.pop('lbl_atk')
        eer, eer_threshold = metrics_dschulz.eer(**pais)
        apcer = metrics_dschulz.apcer(pais['fpr'], pais['thresholds'])
        bpcer = metrics_dschulz.bpcer(pais['fnr'], pais['thresholds'])
        apcer_pais[lbl_atk] = apcer
        eer_pais[lbl_atk] = eer
        bpcer_ap_pais[lbl_atk] = dict()
        for ap in ap_eval:
            bpcer_ap, ap_threshold = metrics_dschulz.bpcer_ap(**{**pais, **{'ap': ap}})
            bpcer_ap_pais[lbl_atk]['bpcer' + str(ap)] = bpcer_ap
        #     print('BPCER' + str(ap) + ': ' + str(bpcer_ap))
        # print('--------------------')
    results['apcer_pais'] = str(apcer_pais)
    results['bpcer'] = bpcer
    worst_pais = max(eer_pais, key=eer_pais.get)
    results['eer'] = eer_pais[worst_pais]
    for key in bpcer_ap_pais[worst_pais]:
        results[key] = bpcer_ap_pais[worst_pais][key]
    # metrics_dschulz.det_curve_new(test_labels, 1 - distances)

    test_feat = model.predict(dataset_test)
    plot_tsne(test_feat, test_labels, labels=display_labels, path_save=os.path.join(results_path, 'tsne_2d_test.jpg'))

    roc_display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    plt.grid()
    plt.title('ROC curve using binary classification')
    plt.savefig(os.path.join(results_path, 'roc_curve.jpg'), dpi=150)

    cm = metrics.confusion_matrix(test_labels, few_shot_predictions)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot()
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(results_path, 'conf_matrix.jpg'), dpi=150)
    plt.close()

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
    # Set seeds for repeatability
    seed = 0
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

    input_shape = params['input_shape']
    batch_size = params['batch_size']
    weights = params['weights']
    epochs = params['epochs']

    exp_dir = datetime.today().strftime('%Y%m%d_%H%M%S') + '_' + socket.gethostname()
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

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
    elif params['backbone'] == 'resnet101':
        backbone_model = resnet.ResNet101(include_top=True, input_shape=input_shape, weights=weights)
        backbone_model = Model(inputs=backbone_model.input, outputs=backbone_model.layers[-2].output)
        preprocessor = resnet.preprocess_input
    elif params['backbone'] == 'resnet152':
        backbone_model = resnet.ResNet152(include_top=True, input_shape=input_shape, weights=weights)
        backbone_model = Model(inputs=backbone_model.input, outputs=backbone_model.layers[-2].output)
        preprocessor = resnet.preprocess_input
    # elif params['backbone'] == 'mobilenetv2_small':
    #     # preprocessor = mobilenet_v3.preprocess_input
    # elif params['backbone'] == 'mobilenetv2_large':
    #     # preprocessor = mobilenet_v3.preprocess_input
    elif params['backbone'] == 'custom':
        try:
            backbone_model = load_model(weights, compile=False)
        except Exception as e:
            sys.exit('If using a custom backbone, corresponding weights must be loaded.')
        backbone_model = Model(inputs=backbone_model.input, outputs=backbone_model.layers[-2].output)
        preprocessor = mobilenet_v2.preprocess_input # !!!!!!!!!!!!!!!!!!!!!!!DEJAR GENERICO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # Add L2 normalization layer
    output = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name='l2_norm')(backbone_model.output)
    model = Model(inputs=backbone_model.input, outputs=output)

    if 'unfreeze_from' in params.keys():
        if params['unfreeze_from'] is not None:
            for layer in model.layers:
                if layer.name == params['unfreeze_from']:
                    break
                else:
                    layer.trainable = False

    if 'path_train' in params.keys():
        dataset_train = dataset_to_dict(params['path_train'])
        if DEBUG:
            random.shuffle(dataset_train)
            dataset_train = dataset_train[0:256]
        n_samples_train = len(dataset_train)
        params['n_samples_train'] = n_samples_train

        if 'oversample' in params.keys():
            if params['oversample'] is True:
                dataset_train = oversample_dataset(dataset_train)

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
        if DEBUG:
            random.shuffle(dataset_val)
            dataset_val = dataset_val[0:256]
        n_samples_val = len(dataset_val)
        params['n_samples_val'] = n_samples_val
        dataset_val = tf.data.Dataset.from_tensor_slices([[x['id'], str(x['label'])] for x in dataset_val])
        dataset_val = dataset_val.map(lambda x: load_image(x, input_shape),
                                      num_parallel_calls=AUTOTUNE).batch(batch_size).map(
            lambda x, y: (preprocessor(x), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    learning_rate = params['optimizer_params']['learning_rate']
    if params['optimizer'] == 'adam':
        optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    elif params['optimizer'] == 'sgd':
        optimizer = tf.keras.optimizers.SGD(lr=learning_rate)

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

    callbacks.append(custom_callbacks.SaveTrainingData(latest_model_path, params))
    callbacks.append(custom_callbacks.SaveTrainingData(best_model_path, params, save_best_only=True, monitor='val_loss'))

    if DEBUG:
        epochs = 10
        warnings.warn('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! WARNING: USING DEBUGGING MODE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    # Train the network
    history = model.fit(
        dataset_train,
        validation_data=dataset_val,
        epochs=epochs,
        callbacks=callbacks)

    # Plot tSNE for training data
    dataset_val = dataset_to_dict(params['path_val'])
    dataset_val = tf.data.Dataset.from_tensor_slices([[x['id'], str(x['label'])] for x in dataset_val])
    dataset_val = dataset_val.map(lambda x: load_image(x, input_shape), num_parallel_calls=AUTOTUNE).batch(batch_size)
    dataset_val = dataset_val.map(lambda x, y: (preprocessor(x), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    val_feat = model.predict(dataset_val)
    val_labels = np.concatenate([y for x, y in dataset_val], axis=0)

    dataset_train = dataset_to_dict(params['path_train'])
    dataset_train = tf.data.Dataset.from_tensor_slices([[x['id'], str(x['label'])] for x in dataset_train])
    dataset_train = dataset_train.map(lambda x: load_image(x, input_shape), num_parallel_calls=AUTOTUNE).batch(batch_size)
    dataset_train = dataset_train.map(lambda x, y: (preprocessor(x), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    train_feat = model.predict(dataset_train)
    train_labels = np.concatenate([(y + np.max(val_labels) + 1) for x, y in dataset_train], axis=0)

    feat = np.concatenate((train_feat, val_feat))
    labels = np.concatenate((train_labels, val_labels))
    disp_labels = sorted(list(set(['val-' + str(x)  for x in val_labels]))) \
                  + sorted(list(set(['train-' + str(x - np.max(val_labels) - 1) for x in train_labels])))

    plot_tsne(feat, labels, labels=disp_labels, path_save=os.path.join(latest_model_path, 'tsne_2d_train_val.jpg'))
    shutil.copy2(os.path.join(latest_model_path, 'tsne_2d_train_val.jpg'), best_model_path)

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
    plt.clf()
