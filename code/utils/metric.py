import numpy as np
import argparse
import json
from PIL import Image
from os.path import join
import torch
from torch import nn
from torch.autograd import Variable
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


def evaluate_model(model, dataset, loss_func, device, num_classes, num_workers, batch_size,crf_cfg):
    # 构建dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    hist_sum = np.zeros((num_classes, num_classes))
    loss_list = []
    model.eval()
    with torch.no_grad():
        for batch, item in enumerate(dataloader):
            if crf_cfg:
                data, ori_data, label = item
                data = data.to(device)
                ori_data = ori_data.to(device)
                label = label.to(device)
                out = model(data, ori_data)
            else:
                data, label = item
                data = data.to(device)
                label = label.to(device)
                out = model(data)

            # 计算loss
            loss = loss_func(out, label)
            loss_list.append(loss.cpu().item())
            # 计算混淆矩阵
            pred = torch.argmax(out, dim=1).cpu().numpy()  # 预测结果
            label = label.cpu().numpy()
            for i in range(len(pred)):
                hist = fast_hist(label[i], pred[i], num_classes)
                hist_sum += hist
    loss = sum(loss_list) / len(loss_list)
    miou = compute_miou(hist_sum)

    return loss, miou


def evaluate_unet3p_model(model, dataset, loss_func, device, num_classes, num_workers=4, batch_size=64):
    # 构建dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    hist_sum = np.zeros((num_classes, num_classes))
    loss_list = []
    model.eval()
    with torch.no_grad():
        for batch, item in enumerate(dataloader):
            data, label = item
            data = data.to(device)
            label = label.to(device)
            out = model(data)
            # 计算loss
            loss = torch.tensor(0, dtype=torch.float32).to(device)
            for pred in out:
                loss += loss_func(pred, label)
            loss_list.append(loss.cpu().item())

            # 计算混淆矩阵
            pred = torch.argmax(out[0], dim=1).cpu().numpy()  # 预测结果
            label = label.cpu().numpy()
            for i in range(len(pred)):
                hist = fast_hist(label[i], pred[i], num_classes)
                hist_sum += hist
    loss = sum(loss_list) / 5.0 / len(loss_list)
    miou = compute_miou(hist_sum)

    return loss, miou


def evaluate_cls_model(model, dataset, loss_func, loss_cls_func, device, num_classes, num_workers=4, batch_size=64):
    # 构建dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    hist_sum = np.zeros((num_classes, num_classes))
    loss_list = []
    loss_cls_list = []
    model.eval()
    with torch.no_grad():
        for batch, item in enumerate(dataloader):
            X, mask, label = item

            batchsize = mask.shape[0]
            classes = 10
            hw = mask.shape[1]

            X = X.to(device)
            mask = mask.to(device)
            label = label.to(device)

            Y, Y_cls = model(X)

            Y_noclass_mask = torch.ones(batchsize * classes, hw, hw).to(device)
            Y_noclass_index = (Y_cls.reshape(-1) <= 0).nonzero(as_tuple=False).squeeze()
            Y_noclass_mask.index_fill_(0, Y_noclass_index, 0)
            Y_noclass_mask = Y_noclass_mask.reshape(batchsize, classes, hw, hw)

            Y = Y * Y_noclass_mask

            # 计算loss
            loss_segm = loss_func(Y, mask)
            loss_cls = loss_cls_func(Y_cls, label.float())

            loss_list.append(loss_segm.cpu().item())
            loss_cls_list.append(loss_cls.cpu().item())
            # 计算混淆矩阵
            pred = torch.argmax(Y, dim=1).cpu().numpy()  # 预测结果
            mask = mask.cpu().numpy()
            for i in range(len(pred)):
                hist = fast_hist(mask[i], pred[i], num_classes)
                hist_sum += hist
    loss = sum(loss_list) / len(loss_list)
    loss_cls = sum(loss_cls_list) / len(loss_cls_list)
    miou = compute_miou(hist_sum)

    return loss, loss_cls, miou
