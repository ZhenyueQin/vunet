from utils.is_metric import get_inception_score
import cv2
import matplotlib.pyplot as plt


def get_all_file_within_a_directory(a_directory, file_type):
    rtn = []
    import glob, os
    os.chdir(a_directory)
    for a_file in glob.glob('*.' + file_type):
        rtn.append(a_file)
    return rtn


def get_all_img_file_within_a_directory(a_directory, img_type):
    rtn = []
    for a_file_name in get_all_file_within_a_directory(a_directory, img_type):
        an_img = cv2.imread(a_file_name)
        an_rgb_img = cv2.cvtColor(an_img, cv2.COLOR_BGR2RGB)
        plt.imshow(an_rgb_img)
        plt.show()

        rtn.append(an_rgb_img)
    return rtn


file_list = get_all_img_file_within_a_directory('/Volumes/Qin-Warehouse/Warehouse-Data/Variational-U-Net/datasets/deepfashion/test', 'jpg')

get_inception_score(file_list)

