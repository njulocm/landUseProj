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
from torch import randperm

from torchvision import transforms as T
import numpy as np

# train_mean = [0, 0, 0, 0]
# train_std = [1, 1, 1]
#
# X = cv2.imread('/home/cm/landUseProj/tcdata/suichang_round1_train_210120/000003.tif', cv2.IMREAD_UNCHANGED)
# transform = T.Compose([T.ToTensor(),
#                        T.Normalize(mean=train_mean, std=train_std),
#                        T.RandomApply(T.RandomVerticalFlip(p=0.5))])
# X1 = transform(X)


# for i,j in tqdm(enumerate(range(100))):
#     time.sleep(2)

# model = PSPNet(layers=50, classes=10,in_chans=4)
# model.train()
# # model = nn.DataParallel(PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50').cuda())
# X = torch.ones((2, 4, 256, 256))
# label = torch.ones((2, 256, 256)).long()
# Y = model(X, label)

cfg = Config.fromfile('config/Unet_config.py')
dataset_cfg = cfg.dataset_cfg

train_cfg = cfg.train_cfg
auto_save_epoch = train_cfg.setdefault(key='auto_save_epoch', default=50)

# land_dataset = LandDataset(dataset_cfg.test_dir,
#                            input_channel=dataset_cfg.input_channel,
#                            transform=dataset_cfg.test_transform)
# xx = land_dataset.__getitem__(index=10)

# check_point_file = '/home/cm/landUseProj/code/checkpoint/NestedUnet11/NestedUnet_model.pth'
# check_point_dir = '/'.join(check_point_file.split('/')[:-1])
# if not os.path.exists(check_point_dir):
#     os.mkdir(check_point_dir)

# dataset = LandDataset(DIR=dataset_cfg.train_dir,
#                       input_channel=4,
#                       transform=dataset_cfg.train_transform)
#
# data,label=dataset.__getitem__(97)

# x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
# x = np.ones((256,256,4))
# y = np.rot90(x)

print('end')
