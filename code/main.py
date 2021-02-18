import sys
from utils import Config
from tools import train_main, test_main

if __name__ == '__main__':
    # 命令行参数
    cfg_filename = sys.argv[1]  # 配置文件名
    mode = sys.argv[2]  # 运行模式，包括train和test

    # 读取配置文件gio
    cfg = Config.fromfile('config/' + cfg_filename)
    print("config filename: " + str(cfg_filename))

    # 训练模型
    if mode == 'train':
        train_main(cfg=cfg)

    # 测试模型
    if mode == 'test':
        test_main(cfg)

    print('end')
    print("this is a test")
