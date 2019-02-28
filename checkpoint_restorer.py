import os
from main_models import Model
import argparse
import yaml
import utilities
from batches import get_batches, plot_batch, postprocess, n_boxes

default_log_dir = os.path.join(os.getcwd(), "restorer_log")
config_path = 'deepfashion_local.yaml'
out_dir, logger = utilities.init_logging(default_log_dir)

model_path = '/Volumes/Qin-Warehouse/Warehouse-Data/Variational-U-Net/log/2019-02-06T18-10-49/checkpoints/model.ckpt-100000'

with open(config_path) as f:
    config = yaml.load(f)

batch_size = config["batch_size"]
img_shape = 2 * [config["spatial_size"]] + [3]
data_index = config["data_index"]
box_factor = config["box_factor"]

data_shape = [batch_size] + img_shape
init_shape = [config["init_batches"] * batch_size] + img_shape
testing_batches = get_batches(data_shape, data_index, train=False, box_factor=box_factor)

print('what is testing batches: ', testing_batches)

model = Model(config, out_dir, logger)
print('restoring the graph ... ')
model.restore_graph(model_path)

batch_idx = 0
while True:
    X_batch, C_batch, XN_batch, CN_batch = next(testing_batches)
    if X_batch is not None:
        test_rsts = model.test(C_batch)

        for k in test_rsts:
            a_value_set = test_rsts[k]

            k_dir = os.path.join(out_dir, k)
            if not os.path.exists(k_dir):
                os.makedirs(k_dir)

            overall_name = os.path.join(k_dir, "bch_{}_test".format(batch_idx))
            utilities.plot_separate_images_in_a_batch(a_value_set, overall_name)
        batch_idx += 1
    else:
        break
