import os
import shutil
import glob
import random

import torch
import tifffile as tif
from scipy import ndimage
import numpy as np

import conf as conf
from conf import PARAMS as params

def _reshape_np(a):
    f, z, c, h, w = a.shape

    target = (c, h, w, z)

    a = np.reshape(a, target)

    return a

def _flatten_list(l):
    return [item for sublist in l for item in sublist]

def get_image_paths(somite_counts, data_dir):

    all_image_paths = []

    for sc in somite_counts:

        folder = "{}{}{}".format(
            (data_dir),
            str(sc),
            '/*.tif'
        )

        folder = glob.glob(folder)

        all_image_paths.extend(folder)

    return all_image_paths


def get_class_folders(img_folder):

    random.shuffle(img_folder)

    ss_17 = []
    ss_21 = []
    ss_25 = []
    ss_29 = []

    ss_folders = {
        '17': ss_17,
        '21': ss_21,
        '25': ss_25,
        '29': ss_29,
    }

    for i in range(len(img_folder)):
        img = img_folder[i]
        img_dir = os.path.dirname(img)[-2:]

        ss_folders[img_dir].append(img)

    return [ss_17, ss_21, ss_25, ss_29]


def test_train_split(data_dir, img_folder, val_ratio, n_classes):
    """

    Split a set of images into training and validation subsets.

    Args:
        img_folder (list): List of all the image paths.
        val_ratio (float): The ratio by which the dataset will be split

    """

    # Ensure uniform sample size for each class in both test and val subsets

    all_folders = [os.path.dirname(i) for i in img_folder]

    class_lengths = [all_folders.count(data_dir + i)
                     for i in conf.SOMITE_COUNTS]

    lowest_class_len = min(class_lengths)

    target_class_size = lowest_class_len if lowest_class_len % 2 == 0 else lowest_class_len - 1

    train_ratio = 1 - val_ratio
    train_class_size = int(target_class_size * train_ratio)

    val_class_size = int(target_class_size * val_ratio)
    train_set = []

    for folder in get_class_folders(img_folder):
        train_set.append(folder[:train_class_size])

    train_set = _flatten_list(train_set)

    val_set = []

    for folder in get_class_folders(img_folder):
        val_set.append(folder[:val_class_size])

    val_set = _flatten_list(val_set)

    return train_set, val_set


def normalize_stack(stack, target_shape):

    # Target dimensions
    t_x = target_shape[1]
    t_y = target_shape[2]
    t_z = target_shape[3]

    # Current dimensions
    x = stack.shape[1]
    y = stack.shape[2]
    z = stack.shape[3]

    # Calculated scale factor between current and target dimensions
    x_sf = t_x / x
    y_sf = t_y / y
    z_sf = t_z / z

    reshaped_stack = ndimage.zoom(stack, (1, x_sf, y_sf, z_sf))

    return reshaped_stack

def weights_init(m):
    if type(m) == torch.nn.Linear:
        m.weight.data.fill_(1.0)    