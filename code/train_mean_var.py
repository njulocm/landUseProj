#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author : Yujian
# @Time : 2021/2/1 6:57 下午
from utils import evaluate_model, LandDataset
import os
import cv2
import numpy as np
from tqdm import tqdm
root_dir = '/home/yujian/landUseProj'
train_dir=root_dir + '/tcdata/suichang_round1_train_210120'

mean = [[],[],[],[]]
std = [[],[],[],[]]

for index in tqdm(range(len(os.listdir(train_dir))//2)):
    filename = train_dir + '/{:0>6d}'.format(index + 1)
    data = cv2.imread(filename + '.tif', cv2.IMREAD_UNCHANGED).transpose((2, 0, 1))
    data = (data[:4] / 255.0).astype(np.float32)
    label = cv2.imread(filename + '.png', cv2.IMREAD_GRAYSCALE) - 1
    label = label.astype(np.int)
    for i in range(4):
        m, s = cv2.meanStdDev(data[i, :, :])
        mean[i].append(m)
        std[i].append(s)
mean = np.array(mean)
std = np.array(std)
print(mean.shape)
m = np.mean(mean, axis=1)
s = np.mean(std, axis=1)
print(np.squeeze(m))
print(np.squeeze(s))

