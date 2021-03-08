from utils import evaluate_model, LandDataset

import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from utils import evaluate_model, evaluate_cls_model, evaluate_unet3p_model, LandDataset, adjust_learning_rate, \
    moving_average, fix_bn, bn_update, dense_crf
import torch.nn.functional as F
from tqdm import tqdm
import ttach as tta
import numpy as np
import cv2
import os


def test_main(cfg):
    # config
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
    
    def _init_fn():
        np.random.seed(cfg.random_seed)

    dataloader = DataLoader(dataset, batch_size=test_cfg.batch_size, shuffle=False, num_workers=test_cfg.num_workers,
                            worker_init_fn=_init_fn())


    is_ensemble = test_cfg.setdefault(key='is_ensemble', default=False)

    if not is_ensemble:  # 没有使用多模型集成
        # 加载模型
        model = torch.load(test_cfg.check_point_file,
                           map_location=device)  # device参数传在里面，不然默认是先加载到cuda:0，to之后再加载到相应的device上

        # 预测结果
        if test_cfg.is_predict:
            predict(model=model, dataset=dataset, out_dir=test_cfg.out_dir, device=device,
                    batch_size=test_cfg.batch_size)

        # 评估模型
        if test_cfg.is_evaluate:
            loss_func = nn.CrossEntropyLoss().to(device)
            evaluate_model(model, dataset, loss_func, device, cfg.num_classes,
                           num_workers=test_cfg.num_workers, batch_size=test_cfg.batch_size)
    else:  # 使用多模型集成
        # 加载多个模型
        models = []
        for ckpt in test_cfg.check_point_file:
            models.append(torch.load(ckpt, map_location=device))

        # 获取模型集成的权重
        ensemble_weight = test_cfg.setdefault(key='ensemble_weight', default=[1.0 / len(models)] * len(models))
        if len(ensemble_weight) != len(models):
            raise Exception('权重个数错误！')

        # 预测结果
        ensemble_predict(models=models,
                         ensemble_weight=ensemble_weight,
                         dataset=dataset,
                         out_dir=test_cfg.out_dir,
                         device=device,
                         batch_size=test_cfg.batch_size)


def predict(model, dataloader, out_dir, device, sample_index_list):
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

    # eval mode
    model.eval()
    with torch.no_grad():
        for batch, item in tqdm(enumerate(dataloader)):
            data, _ = item
            data = data.to(device)
            out = model(data)
            # out = out[0]
            out = sum(out)/len(out)
            pred = torch.argmax(out, dim=1).cpu().numpy()
            for i in range(len(pred)):
                sample_index = sample_index_list[batch * batch_size + i]
                out_name = out_dir + f'/{sample_index}.png'
                cv2.imwrite(out_name, pred[i] + 1)  # 提交的结果需要1~10
    # 把输出结果打成压缩包
    os.system(f'zip {out_dir[:-1]}.zip {out_dir}*')


def ensemble_predict(models, ensemble_weight, dataset, out_dir, device, batch_size=128):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # 获取数据集中样本的序号
    sample_index_list = dataset.index_list

    # 构建dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    with torch.no_grad():
        for batch, item in tqdm(enumerate(dataloader)):
            data, _ = item
            data = data.to(device)
            out_avg = torch.zeros(size=(len(data), 10, 256, 256)).to(device)
            model_num = 0
            for model in models:
                temp_out = model(data)
                out_avg += ensemble_weight[model_num] * temp_out
                model_num += 1
            pred = torch.argmax(out_avg, dim=1).cpu().numpy()
            for i in range(len(pred)):
                sample_index = sample_index_list[batch * batch_size + i]
                out_name = out_dir + f'/{sample_index}.png'
                cv2.imwrite(out_name, pred[i] + 1)  # 提交的结果需要1~10

    # 把输出结果打成压缩包
    os.system(f'zip {out_dir[:-1]}.zip {out_dir}*')

