import os
import logging
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def MatplotlibClearMemory():
    #usedbackend = matplotlib.get_backend()
    #matplotlib.use('Cairo')
    allfignums = plt.get_fignums()
    for i in allfignums:
        fig = plt.figure(i)
        fig.clear()
        plt.close( fig )


def make_dirs(save_dir):
    existing_versions = os.listdir(save_dir)
    
    if  len(existing_versions) > 0:
        max_version = int(existing_versions[0].split("_")[-1])
        for v in existing_versions:
            ver = int(v.split("_")[-1])
            if ver > max_version:
                max_version = ver
        version = int(max_version) + 1
    else:
        version = 0

    return f"{save_dir}/exp_{version}"

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
 
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
 
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
 
    return logger

def make_one_hot(labels, classes):
    one_hot = torch.FloatTensor(labels.size(0), classes, labels.size(2), labels.size(3)).zero_().to(labels.device)
    target = one_hot.scatter_(1, labels.data.type(torch.int64), 1)
    return target

class DiceCoeff(nn.Module):
    def __init__(self, smooth=1.):
        super().__init__()
        self.smooth = smooth

    def forward(self, y_true, y_pred):
        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()
        intersection = (y_true_f * y_pred_f).sum()
        return (2. * intersection + self.smooth) / (y_true_f.sum() + y_pred_f.sum() + self.smooth)

# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import os
import numpy as np
import pickle
import PIL.Image

# import dnnlib.submission.submit as submit

# save_pkl, load_pkl are used by the mri code to save datasets
def save_pkl(obj, filename):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)

def load_pkl(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# save_snapshot, load_snapshot are used save/restore trained networks
def save_snapshot(submit_config, net, fname_postfix):
    dump_fname = os.path.join(submit_config.run_dir, "network_%s.pickle" % fname_postfix)
    with open(dump_fname, "wb") as f:
        pickle.dump(net, f)

# def load_snapshot(fname):
#     fname = os.path.join(submit.get_path_from_template(fname))
#     with open(fname, "rb") as f:
#         return pickle.load(f)


def save_image(submit_config, img_t, filename):
    t = img_t.transpose([1, 2, 0])  # [RGB, H, W] -> [H, W, RGB]
    if t.dtype in [np.float32, np.float64]:
        t = clip_to_uint8(t)
    else:
        assert t.dtype == np.uint8
    PIL.Image.fromarray(t, 'RGB').save(os.path.join(submit_config.run_dir, filename))

def clip_to_uint8(arr):
    return np.clip((arr + 0.5) * 255.0 + 0.5, 0, 255).astype(np.uint8)

def crop_np(img, x, y, w, h):
    return img[:, y:h, x:w]

# Run an image through the network (apply reflect padding when needed
# and crop back to original dimensions.)
def infer_image(net, img):
    w = img.shape[2]
    h = img.shape[1]
    pw, ph = (w+31)//32*32-w, (h+31)//32*32-h
    padded_img = img
    if pw!=0 or ph!=0:
        padded_img  = np.pad(img, ((0,0),(0,ph),(0,pw)), 'reflect')
    inferred = net.run(np.expand_dims(padded_img, axis=0), width=w+pw, height=h+ph)
    return clip_to_uint8(crop_np(inferred[0], 0, 0, w, h))

