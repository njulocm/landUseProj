from model import U_Net, AttU_Net, NestedUNet
import torch
import torch.nn as nn

from model import PSPNet
from utils import LandDataset, Config
import os
from tqdm import tqdm
import time
import logging
import cv2
# from torch import randperm

from torchvision import transforms as T
import numpy as np
import utils.transforms_DL as T_DL

import matplotlib.pyplot as plt


def histeq(img, nbr_bins=256):
    """ Histogram equalization of a grayscale image. """
    # 获取直方图p(r)
    imhist, bins = np.histogram(img.flatten(), nbr_bins, normed=True)
    # 获取T(r)
    cdf = imhist.cumsum()  # cumulative distribution function
    cdf = 255 * cdf / cdf[-1]
    # 获取s，并用s替换原始图像对应的灰度值
    result = np.interp(img.flatten(), bins[:-1], cdf)
    return result.reshape(img.shape)

train_mean = [0, 0, 0, 0]
train_std = [1, 1, 1, 1]

train_dir = "S:/landUseProj/tcdata/suichang_round1_train_210120/"
test_dir = "S:/landUseProj/tcdata/suichang_round1_test_partA_210120/"
img = cv2.imread(test_dir+'001000.tif', cv2.IMREAD_UNCHANGED)
label = cv2.imread('S:/landUseProj/tcdata/suichang_round1_train_210120/000333.png', cv2.IMREAD_GRAYSCALE) - 1

# prob = 0.5
# # T.RandomRotation(90)(torch.ones(1,256,256))
# transform = T.Compose([
#     T_DL.ToTensor_DL(),
#     T_DL.Normalized_DL(mean=train_mean, std=train_std),
#     T_DL.RandomFlip_DL(p=1),
#     T_DL.RandomRotation_DL(p=1),
#     T_DL.RandomColorJitter_DL(p=1, brightness=1, contrast=1, saturation=1, hue=0.5),
#     T_DL.Normalized_DL(mean=train_mean, std=train_std)
# ])
# img1, label1 = transform((img, label))
# label2 = label1.numpy()
# print('end')

img = img[:,:,3]
plt.imshow(img)
plt.show()
plt.hist(img.ravel(), 256, [0, 256])#ravel函数功能是将多维数组降为一维数组
plt.show()
# img = ((np.clip(img,30,230)-30).astype(np.float)*256.0/200.0).astype(np.int64)
# plt.hist(img.ravel(), 256, [0, 256])#ravel函数功能是将多维数组降为一维数组
# plt.show()
# img = histeq(img)
# plt.imshow(img)
# plt.show()
# cv2.waitKey(0)
# cv2.destroyAllWindows()


