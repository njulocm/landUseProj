import numpy as np
import argparse
import json
from PIL import Image
from os.path import join
import torch
from torch import nn
from torch.utils.data import DataLoader


def fast_hist(label, pred, num_classes):
    '''
    计算label和pred的混淆矩阵
    :param label:
    :param pred:
    :param num_classes:
    :return hist: 返回混淆矩阵(num_classes, num_classes)
    '''

    # 先把二维矩阵拉平
    a = label.flatten()
    b = pred.flatten()
    hist = np.bincount(num_classes * a.astype(int) + b, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)  很机智！！！
    return hist


def compute_miou(hist):  # 分别为每个类别（在这里是19类）计算mIoU，hist的形状(n, n)
    '''
    由混淆矩阵计算miou
    :param hist: 混淆矩阵
    :return miou:
    '''
    iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))  # 每个类比的iou，n维向量
    miou = iou.mean()
    return miou


def evaluate_model(model, dataset, loss_func, device, num_classes, num_workers, batch_size=64):
    # 构建dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    hist_sum = np.zeros((num_classes, num_classes))
    loss_list = []
    with torch.no_grad():
        for batch, item in enumerate(dataloader):
            data, label = item
            data = data.to(device)
            label = label.to(device)
            out = model(data)
            # 计算loss
            loss = loss_func(out, label)
            loss_list.append(loss.cpu().item())
            # 计算混淆矩阵
            pred = torch.argmax(out, dim=1).cpu().numpy() # 预测结果
            label = label.cpu().numpy()
            for i in range(len(pred)):
                hist = fast_hist(pred[i], label[i], num_classes)
                hist_sum += hist
    loss = sum(loss_list) / len(loss_list)
    miou = compute_miou(hist_sum)

    return loss, miou

