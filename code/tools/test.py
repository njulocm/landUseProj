from utils import evaluate_model, LandDataset

import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader

import cv2
import os


def test_main(cfg):
    # config
    dataset_cfg = cfg.dataset_cfg
    test_cfg = cfg.test_cfg
    device = cfg.device

    if test_cfg.dataset == 'val_dataset':
        land_dataset = LandDataset(dataset_cfg.train_dir,
                                   input_channel=dataset_cfg.input_channel,
                                   transform=dataset_cfg.train_transform)
        # 划分数据集
        train_size = int(dataset_cfg.train_ratio * len(land_dataset))
        val_size = len(land_dataset) - train_size
        _, dataset = random_split(land_dataset, [train_size, val_size],
                                  generator=torch.manual_seed(dataset_cfg.random_seed))
    elif test_cfg.dataset == 'test_dataset':
        dataset = LandDataset(dataset_cfg.test_dir,
                              input_channel=dataset_cfg.input_channel,
                              transform=dataset_cfg.test_transform)
    else:
        raise Exception('没有配置数据集！')

    # 加载模型
    model = torch.load(test_cfg.check_point_file).to(device)

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
    sample_index_list = dataset.indices

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
                out_name = out_dir + '/{:0>6d}.png'.format(sample_index + 1)
                cv2.imwrite(out_name, pred[i]+1) # 提交的结果需要1~10
