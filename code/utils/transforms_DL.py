# 同时对data和label都进行变换，故加上后缀DL
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F


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


class Normalized_DL(torch.nn.Module):

    def __init__(self, mean, std, inplace=False):
        super().__init__()
        self.normalize = T.Normalize(mean=mean, std=std, inplace=inplace)

    def forward(self, img_label):
        img, label = img_label
        img = self.normalize(img)
        return img, label
