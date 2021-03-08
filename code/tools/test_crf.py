from utils import evaluate_model, LandDataset

import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import ttach as tta
import numpy as np
import cv2
import os


def test_crf_main(cfg):
    # config
    dataset_cfg = cfg.dataset_cfg
    train_cfg = cfg.train_cfg
    test_cfg = cfg.test_cfg
    device = cfg.device

    if test_cfg.dataset == 'val_dataset':
        dataset = LandDataset(DIR=dataset_cfg.val_dir,
                              mode='val',
                              is_crf=dataset_cfg.is_crf,
                              input_channel=dataset_cfg.input_channel,
                              transform=dataset_cfg.val_transform)
    elif test_cfg.dataset == 'test_dataset':
        dataset = LandDataset(dataset_cfg.test_dir,
                              mode='test',
                              is_crf=dataset_cfg.is_crf,
                              input_channel=dataset_cfg.input_channel,
                              transform=dataset_cfg.test_transform)
    else:
        raise Exception('没有配置数据集！')

    # 加载模型,预测结果
    if test_cfg.is_trainingendcrf:
        model_no_crf = torch.load(test_cfg.check_point_file,
                                  map_location=device).to(device)
        model = torch.load(train_cfg.check_point_file,
                           map_location=device).to(device)

        model_no_crf_dict = model_no_crf.state_dict()
        model_dict = model.state_dict()

        model_no_crf_dict = {k: v for k, v in model_no_crf_dict.items() if k in model_dict}
        model_dict.update(model_no_crf_dict)

        model.load_state_dict(model_dict)

    else:
        model = torch.load(test_cfg.check_point_file,
                           map_location=device).to(device)

    # model = model.module  #并行训练的话需要加上这行
    if test_cfg.tta_mode:
        model = tta.SegmentationTTAWrapper(model, tta.aliases.d4_transform(), merge_mode='mean')

    # 预测结果
    if test_cfg.is_predict:
        predict(model=model, dataset=dataset, out_dir=test_cfg.out_dir, device=device,
                crf_cfg=dataset_cfg.is_crf, batch_size=test_cfg.batch_size)

    # 评估模型
    if test_cfg.is_evaluate:
        loss_func = nn.CrossEntropyLoss().to(device)
        evaluate_model(model, dataset, loss_func, device, cfg.num_classes,
                       num_workers=test_cfg.num_workers, batch_size=test_cfg.batch_size)


def predict(model, dataset, out_dir, device, crf_cfg, batch_size=128):
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
    model.eval()
    with torch.no_grad():
        for batch, item in tqdm(enumerate(dataloader)):
            if crf_cfg:
                data, ori_data, label = item
                data = data.to(device)
                ori_data = ori_data.to(device)
                out = model(data, ori_data, mode='test')
            else:
                data, label = item
                data = data.to(device)
                out = model(data, mode='test')

            pred = torch.argmax(out, dim=1).cpu().numpy()
            for i in range(len(pred)):
                sample_index = sample_index_list[batch * batch_size + i]
                out_name = out_dir + f'/{sample_index}.png'
                cv2.imwrite(out_name, pred[i] + 1)  # 提交的结果需要1~10
