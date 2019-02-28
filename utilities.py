import os
import datetime
import shutil
import glob
import logging
import scipy.misc


def init_logging(out_base_dir):
    # get unique output directory based on current time
    os.makedirs(out_base_dir, exist_ok=True)
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out_dir = os.path.join(out_base_dir, now)
    os.makedirs(out_dir, exist_ok=False)
    # copy source code to logging dir to have an idea what the run was about
    this_file = os.path.realpath(__file__)
    assert (this_file.endswith(".py"))
    shutil.copy(this_file, out_dir)
    # copy all py files to logging dir
    src_dir = os.path.dirname(this_file)
    py_files = glob.glob(os.path.join(src_dir, "*.py"))
    for py_file in py_files:
        shutil.copy(py_file, out_dir)
    # init logging
    logging.basicConfig(filename=os.path.join(out_dir, 'log.txt'))
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    return out_dir, logger


def plot_separate_images_in_a_batch(a_batch, save_path):
    for idx in range(len(a_batch)):
        a_file_name = save_path + '_' + 'bch_' + str(idx) + '.png'
        scipy.misc.imsave(a_file_name, a_batch[idx])

