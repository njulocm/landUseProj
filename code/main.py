import sys
from utils import LandDataset, Config, evaluate_model
from tools import train_main, test_main

# from .utils import LandDataset, Config, evaluate_model
# from .tools import train_main, test_main


if __name__ == '__main__':
    # 命令行参数
    cfg_filename = sys.argv[1]  # 配置文件名
    mode = sys.argv[2]  # 运行模式，包括train和test

    # 读取配置文件
    cfg = Config.fromfile('config/' + cfg_filename)

    # 训练模型
    if mode == 'train':
        train_main(cfg=cfg)

    # 测试模型
    if mode == 'test':
        test_main(cfg)

    # model_best = torch.load('/home/cm/landUseProj/code/checkpoint/fcn_model_65.pt').to(cfg.device)
    # loss_func = nn.CrossEntropyLoss().to(cfg.device)
    # result = val_performance = evaluate_model(model, val_dataset, loss_func, cfg.device,
    #                                           cfg.num_classes, cfg.train_cfg.num_workers)

    print('end')
