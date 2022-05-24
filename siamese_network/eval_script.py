import os
import json
from siamese_network import eval_siamese_network
import tensorflow as tf
import time



def main():
    # eval_path = '/home/dschulz/TOC/siameseNetwork/models/finishedTrainingsTripletLoss/test_arg2'
    # eval_path = '/home/dschulz/TOC/fakeid_2.0/siamese_network/20220512_230349_TOC-Desktop-RTX2080Ti/best_model'
    eval_path = '/home/dschulz/TOC/fakeid_2.0/siamese_network/20220517_081718_TOC-Desktop-RTX2080Ti/best_model'
    models_path = list()
    for path, subdirs, files in os.walk(eval_path):
        for name in files:
            if name == 'saved_model.pb':
                models_path.append(path)
    models_path.sort()

    results_dir = 'results'
    results_file = open('results.csv', 'wt', buffering=1)
    delimiter = ';'
    for n, model_path in enumerate(models_path):
        # print(model_path)
        params = None
        if os.path.exists(os.path.join(model_path, 'params.json')):
            with open(os.path.join(model_path, 'params.json')) as f:
                params = json.load(f)
        # print(os.path.join(os.path.split(params['path_val'])[0], 'train.txt'))
        # exit()

        add_params_dict = dict()
        add_params_dict['description'] = params['description'] if 'description' in params.keys() else ''
        add_params_dict['backbone'] = params['backbone']
        # add_params_dict['backbone_params'] = [x + ': '+str(params['backbone_params'][x]) for x in params['backbone_params'].keys()] if 'backbone_params' in params.keys() else ''
        add_params_dict['backbone_params'] = params['backbone_params'] if 'backbone_params' in params.keys() else None
        add_params_dict['data_augmentation'] = params['data_augmentation'] if 'data_augmentation' in params.keys() else None
        add_params_dict['oversample'] = params['oversample'] if 'oversample' in params.keys() else False
        add_params_dict['optimizer'] = params['optimizer']

        results_path = os.path.join(model_path, results_dir, 'results.json')
        if os.path.exists(results_path):
            with open(results_path) as f:
                results = json.load(f)
            # print(results['model_path'], description, backbone, results['auc'])
        else:
            # path_templates = params['path_templates'] if 'path_templates' in params.keys() else params['path_train']
            path_templates = params['path_templates'] if 'path_templates' in params.keys() else os.path.join(os.path.split(params['path_val'])[0], 'train.txt')
            n_templates = params['n_templates'] if 'n_templates' in params.keys() else 4
            display_labels = params['display_labels'] if 'display_labels' in params.keys() else ['d', 'b', 'p', 's']

            eval_siamese_network(model_path, path_templates, params['path_test'], n_templates, display_labels,
                                 os.path.join(model_path, results_dir))

            tf.keras.backend.clear_session()
            with open(results_path) as f:
                results = json.load(f)

        results = {**add_params_dict, **results}
        # Write headers
        if n == 0:
            line = ''
            for key in results.keys():
                line += '%s%s' % (key, delimiter)
            line = line.rstrip(delimiter) + '\n'
            results_file.write(line)

        # Write experiment results
        line = ''
        for key in results.keys():
            line += '%s%s' % (str(results[key]).rstrip('}').lstrip('{'), delimiter)
        line = line.rstrip(delimiter) + '\n'
        results_file.write(line)

    results_file.close()


if __name__ == '__main__':
    main()