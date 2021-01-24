from model import U_Net, AttU_Net, NestedUNet
import torch
import torch.nn as nn

from model import PSPNet
from utils import LandDataset, Config
import os

model = PSPNet(4,10, False, False)
# model = nn.DataParallel(PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50').cuda())
X = torch.ones((2, 4, 256, 256))
Y = model(X)

# cfg = Config.fromfile('config/Unet_config.py')
# dataset_cfg = cfg.dataset_cfg
# land_dataset = LandDataset(dataset_cfg.test_dir,
#                            input_channel=dataset_cfg.input_channel,
#                            transform=dataset_cfg.test_transform)
# xx = land_dataset.__getitem__(index=10)

# check_point_file = '/home/cm/landUseProj/code/checkpoint/NestedUnet11/NestedUnet_model.pth'
# check_point_dir = '/'.join(check_point_file.split('/')[:-1])
# if not os.path.exists(check_point_dir):
#     os.mkdir(check_point_dir)

print('end')
