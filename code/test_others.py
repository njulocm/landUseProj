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

