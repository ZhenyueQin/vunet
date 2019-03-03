import os, logging, shutil, datetime

os.system('nvcc --version')

import glob
import argparse
import yaml

from batches import get_batches, plot_batch, postprocess, n_boxes
import utilities


if __name__ == "__main__":
    default_log_dir = os.path.join(os.getcwd(), "log")

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='path to config')
    parser.add_argument('--mode', default='train',
                        choices=['train', 'test', 'add_reconstructions', 'transfer'])
    parser.add_argument('--log_dir', default=default_log_dir, help='path to log into')
    parser.add_argument('--checkpoint', help='path to checkpoint to restore')
    parser.add_argument('--retrain', dest='retrain', action='store_true', help='reset global_step to zero')
    parser.add_argument('--likelihood_loss',
                        choices=['l1', 'vgg_perception'])
    parser.add_argument('--to_use_imgn', default='true')
    parser.add_argument('--to_use_mean')
    parser.add_argument('--loss_test')
    parser.add_argument('--copy')
    parser.set_defaults(retrain=False)

    opt = parser.parse_args()

    with open(opt.config) as f:
        config = yaml.load(f)

    out_dir, logger = utilities.init_logging(opt.log_dir)
    logger.info(opt)
    logger.info(yaml.dump(config))

    print('out dir: ', out_dir)

    if opt.mode == 'train':
        batch_size = config['batch_size']
        img_shape = 2 * [config['spatial_size']] + [3]
        data_shape = [batch_size] + img_shape
        init_shape = [config['init_batches'] * batch_size] + img_shape
        box_factor = config['box_factor']

        data_index = config['data_index']
        batches = get_batches(data_shape, data_index, train=True, box_factor=box_factor)
        init_batches = get_batches(init_shape, data_index, train=True, box_factor=box_factor)
        valid_batches = get_batches(data_shape, data_index, train=False, box_factor=box_factor)
        logger.info('Number of training samples: {}'.format(batches.n))
        logger.info('Number of validation samples: {}'.format(valid_batches.n))

        if opt.copy is None:
            if opt.loss_test is None:
                if opt.to_use_imgn == 'true':
                    from main_models import Model
                    model = Model(config, out_dir, logger, opt)
                elif opt.to_use_imgn == 'false':
                    from main_models_no_imgn import Model
                    model = Model(config, out_dir, logger, opt)
                elif opt.to_use_imgn == 'no_cn':
                    from main_models_no_cn import Model
                    model = Model(config, out_dir, logger, opt)
            else:
                print('playing loss test')
                from main_models_loss_tester import Model
                model = Model(config, out_dir, logger, opt)
        else:
            if opt.copy == 'no_loop':
                print('Without the autoregressively modeled groups')
                from main_models_no_group_loop import Model
                model = Model(config, out_dir, logger, opt)
            elif opt.copy == 'p_q':
                print('Merely populate p with qs')
                from main_models_copy import Model
                model = Model(config, out_dir, logger, opt)

        if opt.checkpoint is not None:
            model.restore_graph(opt.checkpoint)
        else:
            model.init_graph(next(init_batches))
        if opt.retrain:
            model.reset_global_step()
        model.fit(batches, valid_batches)
    else:
        raise NotImplemented()
