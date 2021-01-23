from utils import evaluate_model, LandDataset

import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader

import cv2
import os


def test_main(cfg):
    # config_hrnet
    dataset_cfg = cfg.dataset_cfg
    test_cfg = cfg.test_cfg
    device = cfg.device

    if test_cfg.dataset == 'val_dataset':
        dataset = LandDataset(DIR=dataset_cfg.val_dir,
                              mode='val',
                              input_channel=dataset_cfg.input_channel,
                              transform=dataset_cfg.val_transform)
    elif test_cfg.dataset == 'test_dataset':
        dataset = LandDataset(dataset_cfg.test_dir,
                              mode='test',
                              input_channel=dataset_cfg.input_channel,
                              transform=dataset_cfg.test_transform)
    else:
        raise Exception('没有配置数据集！')

    # 加载模型
    model = torch.load(test_cfg.check_point_file,map_location=device) # device参数传在里面，不然默认是先加载到cuda:0，to之后再加载到相应的device上

    # 预测结果
    if test_cfg.is_predict:
        predict(model=model, dataset=dataset, out_dir=test_cfg.out_dir, device=device, batch_size=test_cfg.batch_size)

    # 评估模型
    if test_cfg.is_evaluate:
        loss_func = nn.CrossEntropyLoss().to(device)
        evaluate_model(model, dataset, loss_func, device, cfg.num_classes,
                       num_workers=test_cfg.num_workers, batch_size=test_cfg.batch_size)


def predict(model, dataset, out_dir, device, batch_size=128):
    '''
    输出预测结果
    :param model:
    :param dataset:
    :param out_dir:
    :param device:
    :return: None
    '''

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # 获取数据集中样本的序号
    sample_index_list = dataset.index_list

    # 构建dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    with torch.no_grad():
        for batch, item in enumerate(dataloader):
            data, _ = item
            data = data.to(device)
            out = model(data)
            pred = torch.argmax(out, dim=1).cpu().numpy()
            for i in range(len(pred)):
                sample_index = sample_index_list[batch * batch_size + i]
                out_name = out_dir + f'/{sample_index}.png'
                cv2.imwrite(out_name, pred[i]+1) # 提交的结果需要1~10
