import torch
from torch.utils.data import random_split

import torchvision.transforms as T

from utils import LandDataset
import cv2
import os
import numpy as np
from tqdm import tqdm


def copy_train_val():
    root_dir = '/home/cm/landUseProj'
    train_dir = root_dir + '/tcdata/suichang_round1_train_210120'
    test_dir = root_dir + '/tcdata/suichang_round1_test_partA_210120'
    train_ratio = 0.8
    random_seed = 999
    train_mean = [0.12115830639715953, 0.13374122921202505, 0.10591787170772765, 0.5273172240088813]
    train_std = [0.06708223199001057, 0.07782029730399954, 0.06915748947925031, 0.16104953671241798]
    test_mean = [0.1070993648354524, 0.10668084722780714, 0.10053905204822813, 0.4039465719983469]
    test_std = [0.0659528024287434, 0.07412065904513164, 0.07394464607772513, 0.1716164042414669]
    train_transform = T.Compose([T.ToTensor(),
                                 T.Normalize(mean=train_mean, std=train_std),
                                 ])
    # 构建数据集
    land_dataset = LandDataset(train_dir,
                               input_channel=4,
                               transform=train_transform)
    # 划分数据集
    train_size = int(train_ratio * len(land_dataset))
    val_size = len(land_dataset) - train_size
    train_dataset, val_dataset = random_split(land_dataset, [train_size, val_size],
                                              generator=torch.manual_seed(random_seed))

    train_indices = train_dataset.indices
    train_indices.sort()
    val_indices = val_dataset.indices
    val_indices.sort()

    # copy train dataset
    for index in train_indices:
        index_str = '{:0>6d}'.format(index + 1)
        os.system(f'cp {train_dir}/{index_str}.tif {root_dir}/tcdata/train/{index_str}.tif')
        os.system(f'cp {train_dir}/{index_str}.png {root_dir}/tcdata/train/{index_str}.png')

    # copy validation dataset
    for index in val_indices:
        index_str = '{:0>6d}'.format(index + 1)
        os.system(f'cp {train_dir}/{index_str}.tif {root_dir}/tcdata/validation/{index_str}.tif')
        os.system(f'cp {train_dir}/{index_str}.png {root_dir}/tcdata/validation/{index_str}.png')

    print('end')


img_w = 256
img_h = 256


def gamma_transform(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)


def random_gamma_transform(img, gamma_vari):
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    return gamma_transform(img, gamma)


def rotate(xb, yb, angle):
    M_rotate = cv2.getRotationMatrix2D((img_w / 2, img_h / 2), angle, 1)
    xb = cv2.warpAffine(xb, M_rotate, (img_w, img_h))
    yb = cv2.warpAffine(yb, M_rotate, (img_w, img_h))
    return xb, yb


def blur(img):
    img = cv2.blur(img, (3, 3));
    return img


def add_noise(img):
    for i in range(200):  # 添加点噪声
        temp_x = np.random.randint(0, img.shape[0])
        temp_y = np.random.randint(0, img.shape[1])
        img[temp_x][temp_y] = 255
    return img


def data_augment(xb, yb):
    # 旋转   --原来的做法有的label被补成了255，估计旋转发生了边缘行的裁剪
    rotate_random_num = np.random.random()
    if rotate_random_num < 0.25:  # 逆时针旋转90°
        xb = np.rot90(xb, k=1)
        yb = np.rot90(yb, k=1)
    elif rotate_random_num < 0.5: # 逆时针旋转180°
        xb = np.rot90(xb, k=2)
        yb = np.rot90(yb, k=2)
    elif rotate_random_num < 0.75: # 逆时针旋转270°
        xb = np.rot90(xb, k=3)
        yb = np.rot90(yb, k=3)
    else:  # 不旋转
        pass

    # 翻转
    if np.random.random() < 0.25:
        xb = cv2.flip(xb, 1)  # flipcode > 0：沿y轴翻转
        yb = cv2.flip(yb, 1)
    if np.random.random() < 0.25:
        xb = cv2.flip(xb, 0)  # flipcode == 0：沿x轴翻转
        yb = cv2.flip(yb, 0)

    # gamma变换
    if np.random.random() < 0.25:
        xb = random_gamma_transform(xb, 2)

    # 模糊
    if np.random.random() < 0.25:
        xb = blur(xb)

    # 噪声
    if np.random.random() < 0.2:
        xb = add_noise(xb)

    return xb, yb


def argumeant_main(num_arg_each=7):
    train_dir = '/home/cm/landUseProj/tcdata/train/'
    argument_dir = '/home/cm/landUseProj/tcdata/argument/'
    file_list = os.listdir(train_dir)
    file_list.sort()

    for i in tqdm(range(len(file_list) // 2)):
        # 读取data和label
        index = 2 * i
        pic_num_str = file_list[index].split('.')[0]
        data_filename = train_dir + pic_num_str + '.tif'
        label_filename = train_dir + pic_num_str + '.png'
        data = cv2.imread(data_filename, cv2.IMREAD_UNCHANGED)
        label = cv2.imread(label_filename, cv2.IMREAD_GRAYSCALE)

        # 保存原始图片
        cv2.imwrite(argument_dir + pic_num_str + '.tif', data)
        cv2.imwrite(argument_dir + pic_num_str + '.png', label)
        count = 1
        # 増广图片
        while (count <= num_arg_each):
            data_arg, label_arg = data_augment(data, label)
            arg_filename = argument_dir + pic_num_str + f'-{count}'
            cv2.imwrite(arg_filename + '.tif', data_arg)
            cv2.imwrite(arg_filename + '.png', label_arg)
            count += 1

    print('end')


if __name__ == '__main__':
    # copy_train_val()
    argumeant_main()
