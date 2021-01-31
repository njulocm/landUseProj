from torch.utils.data import Dataset
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


# def get_transform(transform_cfg):


class LandDataset(Dataset):
    def __init__(self, DIR, input_channel=4, transform=None):
        '''

        :param DIR: 数据集路径
        :param input_channel: 输入取多少个通道，默认取全部4通道
        '''
        self.DIR = DIR
        self.transform = transform
        self.input_channel = input_channel
        self.index_list = self._get_index_list() # 所有数据的index

        # 判断是train还是test的数据(train中包括)
        if self.DIR.find('test') > -1:
            self.mode = 'test'
            self.indices = list(range(self.__len__()))  # 如果是测试集，手动设置self.indices参数，在预测的时候需要用到
        else:
            self.mode = 'train'



    def _get_index_list(self): # 获得所有数据的index
        index_list = [filename.split('.')[0] for filename in os.listdir(self.DIR)]
        index_list = list(set(index_list))
        index_list.sort()
        return index_list

    def __len__(self):
        '''返回数据集大小'''
        return len(self.index_list)

    # ----有transform的版本----
    def __getitem__(self, index):
        '''获得index序号的样本'''
        filename = self.DIR + '/' + self.index_list[index] # 不含后缀
        data = cv2.imread(filename + '.tif', cv2.IMREAD_UNCHANGED)
        data = self.transform(data)

        if self.mode == 'train':
            label = cv2.imread(filename + '.png', cv2.IMREAD_GRAYSCALE) - 1
            label = label.astype(np.int)
        else:
            label = 0  # 测试集没有label，随便给一个

        return data, label


    # ----之前没有transform的版本----
    # def __getitem__(self, index):
    #     '''获得index序号的样本'''
    #     filename = self.DIR + '/{:0>6d}'.format(index + 1)
    #     data = cv2.imread(filename + '.tif', cv2.IMREAD_UNCHANGED).transpose((2, 0, 1))
    #     data = (data[:self.input_channel] / 255.0).astype(np.float32)
    #
    #     if self.mode == 'train':
    #         label = cv2.imread(filename + '.png', cv2.IMREAD_GRAYSCALE) - 1
    #         label = label.astype(np.int)
    #     else:
    #         label = 0  # 测试集没有label，随便给一个
    #
    #     return data, label
