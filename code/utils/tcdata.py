from torch.utils.data import Dataset
import os
import cv2
import numpy as np
import torchvision.transforms as T
from PIL import Image




class LandDataset(Dataset):
    def __init__(self, DIR_list, mode, input_channel=4, transform=T.ToTensor()):
        '''

        :param DIR: 数据集路径
        :param mode: 数据集模式，包括train,val,test
        :param input_channel: 输入取多少个通道，默认取全部4通道
        '''
        self.DIR_list = DIR_list
        self.mode = mode
        self.transform = transform
        self.input_channel = input_channel
        self.index_list = self._get_index_list()  # 所有数据的index，只对初赛的推理阶段有用
        self.filename_list = self._get_filename_list() # 所有数据的filename(包含路径，但不包含后缀)
        # if self.mode != 'test':
        #     self.class_dict = self._get_class_dict()

    def _get_index_list(self):  # 获得所有数据的index
        for DIR in self.DIR_list:
            index_list = [filename.split('.')[0] for filename in os.listdir(DIR)]
            index_list = list(set(index_list))
            index_list.sort()
        return index_list

    def _get_filename_list(self):  # 获得所有数据的filename(包含路径，但不包含后缀)
        filename_list = []
        for DIR in self.DIR_list:
            index_list = [filename.split('.')[0] for filename in os.listdir(DIR)]
            index_list = list(set(index_list))
            index_list.sort()
            temp_filename_list = [DIR + '/' + index for index in index_list]
            filename_list += temp_filename_list
        return filename_list

    def _get_class_dict(self):  # 获得所有数据的index
        class_dict = {}
        for filename in self.filename_list:
            label_filename = self.DIR + '/' + filename  # 不含后缀
            label = cv2.imread(label_filename + '.png', cv2.IMREAD_GRAYSCALE) - 1
            for i in range(0, 10):
                if i == 0:
                    class_dict[label_filename] = []
                if (label == i).sum() > 0:
                    class_dict[label_filename].append(1)
                else:
                    class_dict[label_filename].append(0)
        return class_dict

    def __len__(self):
        '''返回数据集大小'''
        return len(self.filename_list)

    # ----有transform的版本----
    def __getitem__(self, index):
        '''获得index序号的样本'''
        filename = self.filename_list[index] # 不含后缀
        data = np.array(Image.open(filename + '.tif'), dtype=np.uint8) # 改，np.uint8，很关键！！！
        # data = cv2.imread(filename + '.tif', cv2.IMREAD_UNCHANGED)[..., :self.input_channel]
        # data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        if self.mode == 'test':
            mask = 0  # 测试集没有label，随便给一个
            data = self.transform(data)
            return data, mask
        else:
            mask = cv2.imread(filename + '.png', cv2.IMREAD_GRAYSCALE) - 1
            data, mask = self.transform((data, mask))
            # label = np.array(self.class_dict[filename]).astype(np.uint8)
            # return data, mask, label
            return data, mask.long()

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
