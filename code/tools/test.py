from utils import evaluate_model, LandDataset

import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from utils import evaluate_model, LandDataset, fast_hist, compute_miou
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import cv2
import os
import joblib
import xgboost


def test_main(cfg):
    # config
    dataset_cfg = cfg.dataset_cfg
    test_cfg = cfg.test_cfg
    device = cfg.device

    if test_cfg.dataset == 'val_dataset':
        dataset = LandDataset(DIR_list=dataset_cfg.val_dir_list,
                              mode='val',
                              input_channel=dataset_cfg.input_channel,
                              transform=dataset_cfg.val_transform)
    elif test_cfg.dataset == 'test_dataset':
        dataset = LandDataset(DIR_list=dataset_cfg.test_dir_list,
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
    boost_type = test_cfg.setdefault(key='boost_type', default=None)

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

        if boost_type is None:  # 采用加权平均集成
            # 获取模型集成的权重
            ensemble_weight = test_cfg.setdefault(key='ensemble_weight', default=[1.0 / len(models)] * len(models))
            if len(ensemble_weight) != len(models):
                raise Exception('权重个数错误！')

            if test_cfg.is_evaluate:  # 评估模式
                miou = ensemble_evaluate(models=models,
                                         dataloader=dataloader,
                                         ensemble_weight=ensemble_weight,
                                         device=device,
                                         num_classes=cfg.num_classes)
                print('miou is : {:.4f}'.format(miou))
                return

            # 预测结果
            ensemble_predict(models=models,
                             ensemble_weight=ensemble_weight,
                             dataset=dataset,
                             out_dir=test_cfg.out_dir,
                             device=device,
                             batch_size=test_cfg.batch_size)
        else:  # 采用boost集成
            if boost_type == 'adaBoost':  # 采用adaBoost集成
                boost_model = joblib.load(test_cfg.boost_ckpt_file)
            elif boost_type == 'XGBoost':  # 采用XGBoost集成
                boost_model = xgboost.Booster(model_file=test_cfg.boost_ckpt_file)

            if test_cfg.is_evaluate:
                miou = ensemble_boost_evaluate(models=models,
                                               dataloader=dataloader,
                                               device=device,
                                               num_classes=cfg.num_classes,
                                               boost_type=boost_type,
                                               boost_model=boost_model)
                print('miou is : {:.4f}'.format(miou))
                return

            # 预测结果
            ensemble_boost_predict(models=models,
                                   boost_model=boost_model,
                                   boost_type=boost_type,
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
            out = sum(out) / len(out)
            pred = torch.argmax(out, dim=1).cpu().numpy()
            for i in range(len(pred)):
                sample_index = sample_index_list[batch * dataloader.batch_size + i]
                out_name = out_dir + f'/{sample_index}.png'
                cv2.imwrite(out_name, pred[i] + 1)  # 提交的结果需要1~10
    # 把输出结果打成压缩包
    # os.system(f'zip {out_dir[:-1]}.zip {out_dir}*')


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
                temp_out = torch.nn.functional.softmax(temp_out, dim=1)  # 转成概率
                out_avg += ensemble_weight[model_num] * temp_out
                model_num += 1
            pred = torch.argmax(out_avg, dim=1).cpu().numpy()
            for i in range(len(pred)):
                sample_index = sample_index_list[batch * batch_size + i]
                out_name = out_dir + f'/{sample_index}.png'
                cv2.imwrite(out_name, pred[i] + 1)  # 提交的结果需要1~10

    # 把输出结果打成压缩包
    # os.system(f'zip {out_dir[:-1]}.zip {out_dir}*')


def ensemble_boost_predict(models, boost_model, boost_type, dataset, out_dir, device, batch_size=128):
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
            out_all = None
            for model in models:
                temp_out = model(data)
                temp_out = torch.nn.functional.softmax(temp_out, dim=1)  # 转成概率
                temp_out = torch.unsqueeze(temp_out, dim=1)  # batch*model*chnl*w*h
                if out_all is None:
                    out_all = temp_out
                else:
                    out_all = torch.cat((out_all, temp_out), dim=1)  # batch*model*chnl*w*h
            out_all = out_all.permute(0, 3, 4, 1, 2)  # batch*w*h*model*chnl
            out_all = out_all.reshape((-1, out_all.shape[-2] * out_all.shape[-1]))  # (batch*w*h)*(model*chnl)
            out_all = out_all.cpu().numpy()
            if boost_type == 'adaBoost':
                pred = boost_model.predict(out_all)  # (batch*w*h)
            elif boost_type == 'XGBoot':
                pred = boost_model.predict(xgboost.DMatrix(data=out_all))  # (batch*w*h)
            pred = pred.reshape((-1, 256, 256))  # batch*w*h
            for i in range(len(pred)):
                sample_index = sample_index_list[batch * batch_size + i]
                out_name = out_dir + f'/{sample_index}.png'
                cv2.imwrite(out_name, pred[i] + 1)  # 提交的结果需要1~10

    # 把输出结果打成压缩包
    # os.system(f'zip {out_dir[:-1]}.zip {out_dir}*')


def ensemble_evaluate(models, dataloader, ensemble_weight, device, num_classes=10):
    hist_sum = np.zeros((num_classes, num_classes))
    with torch.no_grad():
        for batch, item in tqdm(enumerate(dataloader)):
            data, label = item
            data = data.to(device)
            print(data)
            out_avg = torch.zeros(size=(len(data), 10, 256, 256)).to(device)
            model_num = 0
            for model in models:
                temp_out = model(data)
                temp_out = torch.nn.functional.softmax(temp_out, dim=1)  # 转成概率
                out_avg += ensemble_weight[model_num] * temp_out
                model_num += 1
                print(sum(sum(torch.argmax(temp_out, dim=1)[0] + 1 == 2)) / 256 / 256)
            pred = torch.argmax(out_avg, dim=1).cpu().numpy()
            label = label.cpu().numpy()
            for i in range(len(pred)):
                hist = fast_hist(label[i], pred[i], num_classes)
                hist_sum += hist
    miou = compute_miou(hist_sum)
    return miou


def ensemble_boost_evaluate(models, dataloader, device, num_classes=10, boost_type=None, boost_model=None):
    hist_sum = np.zeros((num_classes, num_classes))
    with torch.no_grad():
        for batch, item in tqdm(enumerate(dataloader)):
            data, label = item
            data = data.to(device)
            out_all = None
            for model in models:
                temp_out = model(data)
                temp_out = torch.nn.functional.softmax(temp_out, dim=1)  # 转成概率
                temp_out = torch.unsqueeze(temp_out, dim=1)  # batch*model*chnl*w*h
                if out_all is None:
                    out_all = temp_out
                else:
                    out_all = torch.cat((out_all, temp_out), dim=1)  # batch*model*chnl*w*h

            # 把模型输出转成预测的标签pred
            # out_all = torch.argmax(out_all, dim=2)
            # out_all = out_all.permute(0, 2, 3, 1)
            # out_all = out_all.reshape((-1, 14))
            out_all = out_all.permute(0, 3, 4, 1, 2)  # batch*w*h*model*chnl
            out_all = out_all.reshape((-1, out_all.shape[-2] * out_all.shape[-1]))  # (batch*w*h)*(model*chnl)
            out_all = out_all.cpu().numpy()
            if boost_type == 'adaBoost':
                pred = boost_model.predict(out_all)  # (batch*w*h)
            elif boost_type == 'XGBoost':
                pred = boost_model.predict(xgboost.DMatrix(data=out_all))  # (batch*w*h)
            pred = pred.reshape((-1, 256, 256)).astype(np.int64)  # batch*w*h

            label = label.cpu().numpy()
            for i in range(len(pred)):
                hist = fast_hist(label[i], pred[i], num_classes)
                hist_sum += hist
    miou = compute_miou(hist_sum)
    return miou
