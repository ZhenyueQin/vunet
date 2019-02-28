import os
from main_models import Model
import argparse
import yaml
import utilities
from batches import get_batches, plot_batch, postprocess, n_boxes
import datetime
import numpy as np
from tensorflow.python.client import device_lib


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


available_gpus = get_available_gpus()
print('GPUs: ', available_gpus)

if available_gpus:
    to_use_cluster = True
else:
    to_use_cluster = False

default_log_dir = os.path.join(os.getcwd(), "restorer_log")
out_dir, logger = utilities.restorer_init_logging(default_log_dir)

if to_use_cluster:
    model_path = '/flush2/liu162/Zhenyue-Qin/Project-Wukong-Data-Warehouse/deepfashion/checkpoints/model.ckpt-100000'
    config_path = 'deepfashion.yaml'
else:
    model_path = '/Volumes/Qin-Warehouse/Warehouse-Data/Variational-U-Net/log/2019-02-06T18-10-49/checkpoints/model.ckpt-100000'
    config_path = 'deepfashion_local.yaml'


with open(config_path) as f:
    config = yaml.load(f)

batch_size = config["batch_size"]
img_shape = 2 * [config["spatial_size"]] + [3]
data_index = config["data_index"]
box_factor = config["box_factor"]

data_shape = [batch_size] + img_shape
init_shape = [config["init_batches"] * batch_size] + img_shape
testing_batches = get_batches(data_shape, data_index, train=False, box_factor=box_factor, shuffle=False)


model = Model(config, out_dir, logger)
print('restoring the graph ... ')
model.restore_graph(model_path)


def restore_launch(mission_type, bch_limit=None):
    print('mission type: ', mission_type)
    startingDT = datetime.datetime.now()
    print('Starting DT: ', startingDT)
    X_batch_init, C_batch_init, XN_batch_init, CN_batch_init = next(testing_batches)
    plot_batch(X_batch_init, os.path.join(out_dir, 'target_appearance.png'))
    plot_batch(C_batch_init, os.path.join(out_dir, 'target_pose.png'))

    batch_idx = 0
    while True:
        print('current batch idx: ', batch_idx)
        X_batch, C_batch, XN_batch, CN_batch = next(testing_batches)
        if (X_batch is not None and bch_limit is None) or (batch_idx < bch_limit):
            if mission_type == 'test':
                test_rsts = model.test(C_batch)

                for k in test_rsts:
                    a_value_set = test_rsts[k]
                    if k == 'test_sample':

                        k_dir = os.path.join(out_dir, k)
                        if not os.path.exists(k_dir):
                            os.makedirs(k_dir)

                        overall_name = os.path.join(k_dir, "bch_{}_{}".format(batch_idx, mission_type))
                        utilities.save_batch_img_np_txt(a_value_set, overall_name)
                    elif k == 'cond':
                        k_dir = os.path.join(out_dir, k)
                        if not os.path.exists(k_dir):
                            os.makedirs(k_dir)

                        cond_bch_name = k_dir + os.sep + 'bch_{}_{}'.format(batch_idx, mission_type) + '.png'
                        plot_batch(a_value_set, cond_bch_name)

            elif mission_type == 'transfer':
                test_rsts = []
                transfer_rts = model.transfer(XN_batch, CN_batch, C_batch_init)
                bs = X_batch.shape[0]
                for j in range(bs):
                    test_rsts.append(transfer_rts[j, ...])

                overall_name = os.path.join(out_dir, "bch_{}_{}".format(batch_idx, mission_type))
                plot_batch(X_batch, overall_name + '_src_app.png')
                test_rsts = np.array(test_rsts)
                utilities.save_batch_img_np_txt(test_rsts, overall_name)
            batch_idx += 1
        else:
            break
    endingDT = datetime.datetime.now()
    print('ending DT: ', endingDT)


restore_launch('transfer')
restore_launch('test')
