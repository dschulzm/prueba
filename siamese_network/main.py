from siamese_network import train_siamese_network


def main():
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
    params['batch_size'] = 64
    params['epochs'] = 2
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