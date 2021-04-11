import sys
from utils import Config, split_train_val
from tools import train_main, test_main, test_online_main
import random
import numpy as np
import torch
import os


def set_seed(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':


    # 命令行参数
    cfg_filename = sys.argv[1]  # 配置文件名
    mode = sys.argv[2]  # 运行模式，包括train,test,online_test

    # 读取配置文件
    cfg = Config.fromfile(cfg_filename)
    print("config filename: " + str(cfg_filename))

    set_seed(cfg.random_seed)

    # 划分数据集
    if mode == 'split_data':
        split_train_val()

    # 训练模型
    if mode == 'train':
        train_main(cfg=cfg)

    # 测试模型
    if mode == 'test':
        test_main(cfg=cfg)

    # 复赛线上测试
    if mode == 'test_online':
        test_online_main(cfg=cfg)

    print('end')
