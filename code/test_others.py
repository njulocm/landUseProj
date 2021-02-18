from model import U_Net, AttU_Net, NestedUNet, UnetCRF
import torch
import torch.nn as nn

from model import PSPNet
from utils import LandDataset, Config
import os
from tqdm import tqdm
import time
import logging
import cv2
from torch import randperm

from torchvision import transforms as T
import numpy as np
import utils.transforms_DL as T_DL

from model.crfasrnn.crfasrnn_model import CrfRnnNet




# train_mean = [0, 0, 0, 0]
# train_std = [1, 1, 1, 1]
# img = cv2.imread('/home/cm/landUseProj/tcdata/suichang_round1_train_210120/000033.tif', cv2.IMREAD_UNCHANGED)
# label = cv2.imread('/home/cm/landUseProj/tcdata/suichang_round1_train_210120/000333.png', cv2.IMREAD_GRAYSCALE) - 1
# prob = 0.5
# # T.RandomRotation(90)(torch.ones(1,256,256))
# transform = T.Compose([
#     T_DL.ToTensor_DL(),
#     T_DL.Normalized_DL(mean=train_mean, std=train_std),
#     T_DL.RandomFlip_DL(p=prob),
#     T_DL.RandomRotation_DL(p=prob),
#     T_DL.RandomColorJitter_DL(p=prob, brightness=1, contrast=1, saturation=1, hue=0.5),
#     T_DL.Normalized_DL(mean=train_mean, std=train_std)
# ])
# img1, label1 = transform((img, label))
# img2 = img1.numpy()
# label2 = label1.numpy()

data = torch.ones((64, 4, 256, 256))
model = UnetCRF(in_ch=4, out_ch=10, num_iterations=10, crf_init_params=None)
pred = model.forward(data)
train_mean = [0, 0, 0, 0]
train_std = [1, 1, 1, 1]

img = cv2.imread('/home/cm/landUseProj/tcdata/suichang_round1_train_210120/000033.tif', cv2.IMREAD_UNCHANGED)
label = cv2.imread('/home/cm/landUseProj/tcdata/suichang_round1_train_210120/000333.png', cv2.IMREAD_GRAYSCALE) - 1
prob = 0.5
# T.RandomRotation(90)(torch.ones(1,256,256))

transform = T.Compose([
    T_DL.ToTensor_DL(),
    T_DL.Normalized_DL(mean=train_mean, std=train_std),
    T_DL.RandomFlip_DL(p=1),
    T_DL.RandomRotation_DL(p=1),
    T_DL.RandomColorJitter_DL(p=1, brightness=1, contrast=1, saturation=1, hue=0.5),
    T_DL.Normalized_DL(mean=train_mean, std=train_std)
])
img1, label1 = transform((img, label))
label2 = label1.numpy()

print('end')
