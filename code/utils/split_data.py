import torch
from torch.utils.data import random_split

import torchvision.transforms as T

import sys
sys.path.append("..")

from utils import LandDataset
import os

def split_train_val():
    train_dir_list = ['../tcdata/suichang_round1_train_210120', '../tcdata/suichang_round2_train_210316']
    split_val_from_train_ratio = 0.1
    random_seed = 6666
    train_mean = [0.12115830639715953, 0.13374122921202505, 0.10591787170772765, 0.5273172240088813]
    train_std = [0.06708223199001057, 0.07782029730399954, 0.06915748947925031, 0.16104953671241798]

    train_transform = T.Compose([T.ToTensor(),
                                 T.Normalize(mean=train_mean, std=train_std),
                                 ])
    # 构建数据集
    land_dataset = LandDataset(DIR_list=train_dir_list,
                                mode='train',
                                input_channel=4,
                                transform=train_transform)
    # 划分数据集
    val_size = int(len(land_dataset) * split_val_from_train_ratio)
    train_size = len(land_dataset) - val_size

    train_dataset, val_dataset = random_split(land_dataset, [train_size, val_size],
                                              generator=torch.manual_seed(random_seed))

    train_indices = train_dataset.indices
    train_indices.sort()
    val_indices = val_dataset.indices
    val_indices.sort()

    train_filename_list = [land_dataset.filename_list[i] for i in train_indices]
    val_filename_list = [land_dataset.filename_list[i] for i in val_indices]

    # copy train dataset # 全数据跳过
    print("划分train...")
    # for filename in train_filename_list:
    #     os.system(f'cp {filename}.tif ../../tcdata/round2_train/{"/".join(filename.split("/")[-2:])}.tif')
    #     os.system(f'cp {filename}.png ../../tcdata/round2_train/{"/".join(filename.split("/")[-2:])}.png')

    # copy validation dataset
    print("划分val...")
    if not os.path.exists('../user_data/round2_val'):
        os.system('mkdir ../user_data/round2_val')
        os.system('mkdir ../user_data/round2_val/suichang_round1_train_210120')
        os.system('mkdir ../user_data/round2_val/suichang_round2_train_210316')

    for filename in val_filename_list:
        os.system(f'cp {filename}.tif ../user_data/round2_val/{"/".join(filename.split("/")[-2:])}.tif')
        os.system(f'cp {filename}.png ../user_data/round2_val/{"/".join(filename.split("/")[-2:])}.png')

    print('end')


if __name__ == "__main__":
    # split_data() # 初赛使用的
    split_train_val()


# --------------初赛用的划分数据集-----------------------
# def copy_data(train_indices, val_indices, train_dir_name='train', val_dir_name='validation'):
#     if not os.path.exists(f'{root_dir}/tcdata/{train_dir_name}/'):
#         os.mkdir(f'{root_dir}/tcdata/{train_dir_name}/')
#     if not os.path.exists(f'{root_dir}/tcdata/{val_dir_name}/'):
#         os.mkdir(f'{root_dir}/tcdata/{val_dir_name}/')
#
#     # copy train dataset
#     for index in tqdm(train_indices):
#         index_str = '{:0>6d}'.format(index + 1)
#         os.system(f'cp {train_dir}/{index_str}.tif {root_dir}/tcdata/{train_dir_name}/{index_str}.tif')
#         os.system(f'cp {train_dir}/{index_str}.png {root_dir}/tcdata/{train_dir_name}/{index_str}.png')
#
#     # copy validation dataset
#     for index in tqdm(val_indices):
#         index_str = '{:0>6d}'.format(index + 1)
#         os.system(f'cp {train_dir}/{index_str}.tif {root_dir}/tcdata/{val_dir_name}/{index_str}.tif')
#         os.system(f'cp {train_dir}/{index_str}.png {root_dir}/tcdata/{val_dir_name}/{index_str}.png')


# def split_data():
#     root_dir = '/home/cm/landUseProj'
#     train_dir = root_dir + '/tcdata/suichang_round1_train_210120'
#     test_dir = root_dir + '/tcdata/suichang_round1_test_partA_210120'
#     train_ratio = 0.8
#     random_seed = 999
#     train_mean = [0.12115830639715953, 0.13374122921202505, 0.10591787170772765, 0.5273172240088813]
#     train_std = [0.06708223199001057, 0.07782029730399954, 0.06915748947925031, 0.16104953671241798]
#     test_mean = [0.1070993648354524, 0.10668084722780714, 0.10053905204822813, 0.4039465719983469]
#     test_std = [0.0659528024287434, 0.07412065904513164, 0.07394464607772513, 0.1716164042414669]
#     train_transform = T.Compose([T.ToTensor(),
#                                  T.Normalize(mean=train_mean, std=train_std),
#                                  ])
#     # 构建数据集
#     land_dataset = LandDataset(train_dir,
#                                mode='train',
#                                input_channel=4,
#                                transform=train_transform)
#     # 划分数据集
#     train_size = int(train_ratio * len(land_dataset))
#     val_size = len(land_dataset) - train_size
#     train_dataset, val_dataset = random_split(land_dataset, [train_size, val_size],
#                                               generator=torch.manual_seed(random_seed))
#
#     train_indices = train_dataset.indices  # 只有80%
#     val_indices = val_dataset.indices  # 20%
#     train_num_part = len(train_indices) // 4
#     indices_list = []
#     indices_list.append(train_indices[:train_num_part])
#     indices_list.append(train_indices[train_num_part:train_num_part * 2])
#     indices_list.append(train_indices[train_num_part * 2:train_num_part * 3])
#     indices_list.append(train_indices[train_num_part * 3:])
#     indices_list.append(val_indices)
#
#     for i in range(4):
#         temp_train_indices = []
#         temp_val_indices = []
#         for j in range(5):
#             if j == i:
#                 temp_val_indices += indices_list[j]
#             else:
#                 temp_train_indices += indices_list[j]
#         copy_data(train_indices=temp_train_indices,
#                   val_indices=temp_val_indices,
#                   train_dir_name=f'train{i + 1}',
#                   val_dir_name=f'validation{i + 1}')
#
#     print('end')