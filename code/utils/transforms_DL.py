# 同时对data和label都进行变换，故加上后缀DL
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image
import os
import random
import cv2


class ToTensor_DL(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img_label):
        img, label = img_label
        img = F.to_tensor(img)
        label = torch.tensor(label.astype(np.int))
        return img, label


class RandomHorizontalFlip_DL(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img_label):
        img, label = img_label
        if torch.rand(1) < self.p:
            # print('水平翻转')
            img = F.hflip(img)
            label = F.hflip(label)
        return img, label


class RandomVerticalFlip_DL(torch.nn.Module):

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img_label):
        img, label = img_label
        if torch.rand(1) < self.p:
            # print('垂直翻转')
            img = F.vflip(img)
            label = F.vflip(label)
        return img, label


class RandomFlip_DL(torch.nn.Module):

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img_label):
        img, label = img_label
        if torch.rand(1) < self.p:
            # print('水平或垂直翻转')
            random_num = torch.rand(1)
            if random_num < 0.5:  # 水平翻转
                img = F.hflip(img)
                label = F.hflip(label)
            else:  # 垂直翻转
                img = F.vflip(img)
                label = F.vflip(label)
        return img, label


class RandomRotation_DL(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img_label):
        img, label = img_label
        if torch.rand(1) < self.p:
            # print('随机旋转')
            random_num = torch.rand(1)
            if random_num < 0.33:
                angel = 90
            elif random_num < 0.66:
                angel = 180
            else:
                angel = 270
            img = F.rotate(img, angel)
            label = F.rotate(torch.unsqueeze(label, 0), angel)[0]  # 必须是三维才能转
        return img, label


class RandomColorJitter_DL(torch.nn.Module):
    '''
    随机进行颜色抖动，亮度、对比度、饱和度和色调同时变化
    '''

    def __init__(self, p=0.5, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__()
        self.p = p
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.colorJitter = T.ColorJitter(brightness=self.brightness,
                                         contrast=self.contrast,
                                         saturation=self.saturation,
                                         hue=self.hue)

    def forward(self, img_label):
        img, label = img_label
        if torch.rand(1) < self.p:
            # print('图像rgb变换')
            rgb = img[0:3]
            rgb = self.colorJitter(rgb)

            if img.shape[0] > 3:
                nir = img[3]
                img = torch.cat([rgb, torch.unsqueeze(nir, 0)])
        return img, label


class RandomChooseColorJitter_DL(torch.nn.Module):
    '''
    随机进行颜色抖动，亮度、对比度、饱和度和色调只会选择其一变化
    '''

    def __init__(self, p=0.5, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__()
        self.p = p
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.colorJitter_list = [
            T.ColorJitter(brightness=self.brightness),
            T.ColorJitter(contrast=self.contrast),
            T.ColorJitter(saturation=self.saturation),
            T.ColorJitter(hue=self.hue),
        ]

    def forward(self, img_label):
        img, label = img_label
        if torch.rand(1) < self.p:
            # print('图像rgb变换')
            rgb = img[0:3]
            color_opt = random.choice(self.colorJitter_list)  # 随机选择一个颜色操作
            rgb = color_opt(rgb)

            if img.shape[0] > 3:
                nir = img[3]
                img = torch.cat([rgb, torch.unsqueeze(nir, 0)])
        return img, label


class Normalized_DL(torch.nn.Module):

    def __init__(self, mean, std, inplace=False):
        super().__init__()
        self.normalize = T.Normalize(mean=mean, std=std, inplace=inplace)

    def forward(self, img_label):
        img, label = img_label
        img = self.normalize(img)
        return img, label


class RandomResizeCrop_DL(torch.nn.Module):
    '''
    先放大，后裁剪
    '''

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img_label):
        img, label = img_label
        if torch.rand(1) < self.p:
            # print("进行放大裁剪")
            # 放大过程
            label = torch.unsqueeze(label, 0)  # 必须是三维才能放大和裁剪
            ratio = 1 + torch.rand(1)  # 放大比例[1,2]
            size = int(256 * ratio)
            img = F.resize(img=img, size=[size, size], interpolation=Image.BILINEAR)
            label = F.resize(img=label, size=[size, size], interpolation=Image.NEAREST)
            # 裁剪过程
            top = int((size - 256) * torch.rand(1))
            left = int((size - 256) * torch.rand(1))
            img = F.crop(img=img, top=top, left=left, height=256, width=256)
            label = F.crop(img=label, top=top, left=left, height=256, width=256)
            label = label[0]
        return img, label


class RandomJointCrop_DL(torch.nn.Module):
    '''
    随机再选3张图拼在一起，然后随机裁剪
    '''

    def __init__(self, p=0.5, input_channel=4, DIR='/home/cm/landUseProj/tcdata/train/'):
        super().__init__()
        self.p = p
        self.input_channel = input_channel
        self.DIR = DIR
        self.index_list = self._get_index_list()  # 所有数据的index

    def _get_index_list(self):  # 获得路径中所有数据的index
        index_list = [filename.split('.')[0] for filename in os.listdir(self.DIR)]
        index_list = list(set(index_list))
        index_list.sort()
        return index_list

    def _get_img_label(self, index):
        filename = self.DIR + '/' + index  # 不含后缀
        img = cv2.imread(filename + '.tif', cv2.IMREAD_UNCHANGED)[..., :self.input_channel]
        # data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        label = cv2.imread(filename + '.png', cv2.IMREAD_GRAYSCALE) - 1
        return img, label

    def _randomJoint(self, img_label):
        '''
        为输入的img_label再选3个拼成一张
        '''
        img_label_list = [img_label]
        sample_index_list = random.sample(self.index_list, 3)
        for index in sample_index_list:
            img_label_list.append(self._get_img_label(index))
        random.shuffle(img_label_list)

        # 拼接图片
        joint_img = np.empty((512, 512, self.input_channel), dtype=img_label[0].dtype)
        joint_label = np.empty((512, 512), dtype=img_label[1].dtype)

        joint_img[0:256, 0:256, :] = img_label_list[0][0]
        joint_label[0:256, 0:256] = img_label_list[0][1]

        joint_img[256:512, 0:256, :] = img_label_list[1][0]
        joint_label[256:512, 0:256] = img_label_list[1][1]

        joint_img[0:256, 256:512, :] = img_label_list[2][0]
        joint_label[0:256, 256:512] = img_label_list[2][1]

        joint_img[256:512, 256:512, :] = img_label_list[3][0]
        joint_label[256:512, 256:512] = img_label_list[3][1]

        return joint_img, joint_label

    def forward(self, img_label):
        img, label = img_label
        if torch.rand(1) < self.p:
            # print("进行拼接和裁剪")
            # 拼接
            joint_img, joint_label = self._randomJoint(img_label)
            # 裁剪
            top = int(256 * torch.rand(1))
            left = int(256 * torch.rand(1))
            img = joint_img[top:top + 256, left:left + 256, :]
            label = joint_label[top:top + 256, left:left + 256]
        return img, label


class RandomResizeCrop_DL(torch.nn.Module):
    '''
    先放大，后裁剪
    '''

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img_label):
        img, label = img_label
        if torch.rand(1) < self.p:
            # print("进行放大裁剪")
            # 放大过程
            label = torch.unsqueeze(label, 0)  # 必须是三维才能放大和裁剪
            ratio = 1 + torch.rand(1)  # 放大比例[1,2]
            size = int(256 * ratio)
            img = F.resize(img=img, size=[size, size], interpolation=Image.BILINEAR)
            label = F.resize(img=label, size=[size, size], interpolation=Image.NEAREST)
            # 裁剪过程
            top = int((size - 256) * torch.rand(1))
            left = int((size - 256) * torch.rand(1))
            img = F.crop(img=img, top=top, left=left, height=256, width=256)
            label = F.crop(img=label, top=top, left=left, height=256, width=256)
            label = label[0]
        return img, label


class RandomJointCrop_DL(torch.nn.Module):
    '''
    随机再选3张图拼在一起，然后随机裁剪
    '''

    def __init__(self, p=0.5, input_channel=4, DIR='/home/cm/landUseProj/tcdata/train/'):
        super().__init__()
        self.p = p
        self.input_channel = input_channel
        self.DIR = DIR
        self.index_list = self._get_index_list()  # 所有数据的index

    def _get_index_list(self):  # 获得路径中所有数据的index
        index_list = [filename.split('.')[0] for filename in os.listdir(self.DIR)]
        index_list = list(set(index_list))
        index_list.sort()
        return index_list

    def _get_img_label(self, index):
        filename = self.DIR + '/' + index  # 不含后缀
        img = cv2.imread(filename + '.tif', cv2.IMREAD_UNCHANGED)[..., :self.input_channel]
        # data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        label = cv2.imread(filename + '.png', cv2.IMREAD_GRAYSCALE) - 1
        return img, label

    def _randomJoint(self, img_label):
        '''
        为输入的img_label再选3个拼成一张
        '''
        img_label_list = [img_label]
        sample_index_list = random.sample(self.index_list, 3)
        for index in sample_index_list:
            img_label_list.append(self._get_img_label(index))
        random.shuffle(img_label_list)

        # 拼接图片
        joint_img = np.empty((512, 512, self.input_channel), dtype=img_label[0].dtype)
        joint_label = np.empty((512, 512), dtype=img_label[1].dtype)

        joint_img[0:256, 0:256, :] = img_label_list[0][0]
        joint_label[0:256, 0:256] = img_label_list[0][1]

        joint_img[256:512, 0:256, :] = img_label_list[1][0]
        joint_label[256:512, 0:256] = img_label_list[1][1]

        joint_img[0:256, 256:512, :] = img_label_list[2][0]
        joint_label[0:256, 256:512] = img_label_list[2][1]

        joint_img[256:512, 256:512, :] = img_label_list[3][0]
        joint_label[256:512, 256:512] = img_label_list[3][1]

        return joint_img, joint_label

    def forward(self, img_label):
        img, label = img_label
        if torch.rand(1) < self.p:
            # print("进行拼接和裁剪")
            # 拼接
            joint_img, joint_label = self._randomJoint(img_label)
            # 裁剪
            top = int(256 * torch.rand(1))
            left = int(256 * torch.rand(1))
            img = joint_img[top:top + 256, left:left + 256, :]
            label = joint_label[top:top + 256, left:left + 256]
        return img, label
