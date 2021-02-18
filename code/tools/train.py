from torch.utils.data.dataloader import DataLoader
from utils import evaluate_model, evaluate_cls_model,evaluate_unet3p_model, LandDataset, adjust_learning_rate
from model import build_model

from pytorch_toolbelt import losses as L
from .losses import DiceLoss, FocalLoss, SoftCrossEntropyLoss

import torch
from torch import nn
from torch.utils.data import random_split, DataLoader
from torch.autograd import Variable
from torch import optim
from torch.cuda.amp import autocast, GradScaler
import time
import os
import logging
from tqdm import tqdm


def train_unet3p_epoch(model, optimizer, lr_scheduler, loss_func, dataloader, epoch, device):
    '''
    :param model:
    :param optimizer:
    :param loss_func:
    :param dataloader:
    :param device:
    :return: 返回该轮训练的平均loss
    '''
    loss_list = []
    model.train()
    # scaler = GradScaler()
    for batch, item in tqdm(enumerate(dataloader)):
        X, label = item
        X = Variable(X.to(device))
        label = Variable(label.to(device))
        with autocast():
            Y = model(X)

            loss = torch.tensor(0, dtype=torch.float32).to(device)
            for pred in Y:
                loss += loss_func(pred, label)
            # loss = loss_func(Y, label)
            loss_list.append(loss.cpu().item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # if lr_scheduler is not None:
        #     lr_scheduler.step(epoch + batch / dataloader.__len__())

        # if lr_scheduler is not None:
        #     lr_scheduler.step(epoch + batch / dataloader.__len__())

    return sum(loss_list) / 5.0 / len(loss_list)


def train_epoch(model, optimizer, lr_scheduler, loss_func, dataloader, epoch, device):
    '''
    :param model:
    :param optimizer:
    :param loss_func:
    :param dataloader:
    :param device:
    :return: 返回该轮训练的平均loss
    '''
    loss_list = []
    model.train()
    # scaler = GradScaler()
    for batch, item in tqdm(enumerate(dataloader)):
        X, label = item
        X = Variable(X.to(device))
        label = Variable(label.to(device))
        with autocast():
            Y = model(X)
            loss = loss_func(Y, label)
            loss_list.append(loss.cpu().item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            # optimizer.zero_grad()

        if lr_scheduler is not None:
            lr_scheduler.step(epoch + batch / dataloader.__len__())

    return sum(loss_list) / len(loss_list)


def train_stage2_epoch(model, model_2, optimizer, lr_scheduler, loss_func, dataloader, epoch, device):
    '''
    :param model:
    :param optimizer:
    :param loss_func:
    :param dataloader:
    :param device:
    :return: 返回该轮训练的平均loss
    '''
    loss_list = []
    model.train()
    scaler = GradScaler()
    model.eval()
    model_2.train()
    for batch, item in tqdm(enumerate(dataloader)):
        X, label = item
        X1 = X[:, :-1, :, :]
        X2 = X[:, -1:, :, :]
        X1 = Variable(X1.to(device))
        X2 = Variable(X2.to(device))
        label = Variable(label.to(device))
        with autocast():
            Y1 = model(X1)
            Y1 = (torch.argmax(Y1, dim=1) + 1) / 10.0
            Y1 = torch.unsqueeze(Y1, 1)

            new_X = Variable(torch.cat((Y1, X2), 1))
            Y2 = model_2(new_X)

            loss = loss_func(Y2, label)
            loss_list.append(loss.cpu().item())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        if lr_scheduler is not None:
            lr_scheduler.step(epoch + batch / dataloader.__len__())

    return sum(loss_list) / len(loss_list)


def train_PSPNet(model, optimizer, aux_weight, dataloader, device):
    '''
    训练PSPNet
    :param model:
    :param optimizer:
    :param aux_weight: 辅助loss的权重
    :param dataloader:
    :param device:
    :return: 返回该轮训练的平均loss
    '''
    loss_list = []
    model.train()
    for batch, item in tqdm(enumerate(dataloader)):
        X, label = item
        X = X.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        Y, main_loss, aux_loss = model(X, label)
        loss = main_loss + aux_weight * aux_loss
        loss.backward()
        optimizer.step()
        loss_list.append(loss.cpu().item())
    return sum(loss_list) / len(loss_list)


def train_main(cfg):
    '''
    训练的主函数
    :param cfg: 配置
    :return:
    '''

    # config
    train_cfg = cfg.train_cfg
    dataset_cfg = cfg.dataset_cfg
    model_cfg = cfg.model_cfg
    device = cfg.device

    # 配置logger
    logging.basicConfig(filename=cfg.logfile,
                        filemode='a',
                        level=logging.INFO,
                        format='%(asctime)s\n%(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()

    #
    # 构建数据集
    train_dataset = LandDataset(DIR=dataset_cfg.train_dir,
                                mode='train',
                                input_channel=dataset_cfg.input_channel,
                                transform=dataset_cfg.train_transform)
    val_dataset = LandDataset(DIR=dataset_cfg.val_dir,
                              mode='val',
                              input_channel=dataset_cfg.input_channel,
                              transform=dataset_cfg.val_transform)

    # 构建dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=train_cfg.batch_size, shuffle=True,
                                  num_workers=train_cfg.num_workers)

    # 构建模型
    model = build_model(model_cfg).to(device)
    # model = torch.nn.DataParallel(model)

    # 定义优化器
    optimizer_cfg = train_cfg.optimizer_cfg
    lr_scheduler_cfg = train_cfg.lr_scheduler_cfg
    if optimizer_cfg.type == 'adam':
        optimizer = optim.Adam(params=model.parameters(),
                               lr=optimizer_cfg.lr,
                               weight_decay=optimizer_cfg.weight_decay)
    elif optimizer_cfg.type == 'adamw':
        optimizer = optim.AdamW(model.parameters(),
                                lr=optimizer_cfg.lr, weight_decay=optimizer_cfg.weight_decay)
    elif optimizer_cfg.type == 'sgd':
        optimizer = optim.SGD(params=model.parameters(),
                              lr=optimizer_cfg.lr,
                              momentum=optimizer_cfg.momentum,
                              weight_decay=optimizer_cfg.weight_decay)
    elif optimizer_cfg.type == 'RMS':
        optimizer = optim.RMSprop(params=model.parameters(), lr=optimizer_cfg.lr,
                                  weight_decay=optimizer_cfg.weight_decay)
    else:
        raise Exception('没有该优化器！')

    if lr_scheduler_cfg.policy == 'cos':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2, eta_min=1e-5,
                                                                            last_epoch=-1)
    elif lr_scheduler_cfg.policy == 'LambdaLR':
        import math
        lf = lambda x: (((1 + math.cos(x * math.pi / train_cfg.num_epochs)) / 2) ** 1.0) * 0.95 + 0.05  # cosine
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        lr_scheduler.last_epoch = 0
    else:
        lr_scheduler = None

    # 定义损失函数
    DiceLoss_fn = DiceLoss(mode='multiclass')
    SoftCrossEntropy_fn = SoftCrossEntropyLoss(smooth_factor=0.1)
    loss_func = L.JointLoss(first=DiceLoss_fn, second=SoftCrossEntropy_fn,
                            first_weight=0.5, second_weight=0.5).cuda()
    # loss_cls_func = torch.nn.BCEWithLogitsLoss()

    # 创建保存模型的文件夹
    check_point_dir = '/'.join(model_cfg.check_point_file.split('/')[:-1])
    if not os.path.exists(check_point_dir):  # 如果文件夹不存在就创建
        os.mkdir(check_point_dir)

    # 开始训练
    auto_save_epoch = train_cfg.setdefault(key='auto_save_epoch', default=5)  # 每隔几轮保存一次模型，默认为5
    is_PSPNet = train_cfg.setdefault(key='is_PSPNet', default=False)  # 是否是训练PSPNet，默认为False
    train_loss_list = []
    val_loss_list = []
    val_loss_min = 999999
    best_epoch = 0
    best_miou = 0
    train_loss = 10  # 设置一个初始值
    logger.info('开始在{}上训练{}模型...'.format(device, model_cfg.type))
    for epoch in range(train_cfg.num_epochs):
        print()
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        start_time = time.time()
        print(f"正在进行第{epoch}轮训练...")
        logger.info('*' * 10 + f"第{epoch}轮" + '*' * 10)
        #
        # 训练一轮
        if is_PSPNet:  # 如果是PSPNet,用不同的训练方式
            train_loss = train_PSPNet(model=model,
                                      optimizer=optimizer,
                                      aux_weight=model_cfg.aux_weight,
                                      dataloader=train_dataloader,
                                      device=device)
        else:  # 普通的训练方式
            # train_loss, train_cls_loss = train_cls_epoch(model, optimizer, lr_scheduler, loss_func, loss_cls_func,
            #                                              train_dataloader,
            #                                              epoch, device)
            # train_loss = train_epoch(model, optimizer, lr_scheduler, loss_func, train_dataloader, epoch, device)
            train_loss = train_unet3p_epoch(model, optimizer, lr_scheduler, loss_func, train_dataloader, epoch, device)
            lr_scheduler.step()

        #
        # 在训练集上评估模型
        val_loss, val_miou = evaluate_unet3p_model(model, val_dataset, loss_func, device,
                                            cfg.num_classes, train_cfg.num_workers, batch_size=train_cfg.batch_size)
        # val_loss, val_miou = evaluate_model(model, val_dataset, loss_func, device,
        #                                     cfg.num_classes, train_cfg.num_workers, batch_size=train_cfg.batch_size)
        # val_loss, val_cls_loss, val_miou = evaluate_cls_model(model, val_dataset, loss_func, loss_cls_func, device,
        #                                                       cfg.num_classes, train_cfg.num_workers,
        #                                                       batch_size=train_cfg.batch_size)



        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        # 保存模型
        if val_loss < val_loss_min:
            val_loss_min = val_loss
            best_epoch = epoch
            best_miou = val_miou
            torch.save(model, model_cfg.check_point_file)

        if epoch % auto_save_epoch == auto_save_epoch - 1:  # 每auto_save_epoch轮保存一次
            model_file = model_cfg.check_point_file.split('.')[0] + '-epoch{}.pth'.format(epoch)
            torch.save(model, model_file)

        # 打印中间结果
        end_time = time.time()
        run_time = int(end_time - start_time)
        m, s = divmod(run_time, 60)
        time_str = "{:02d}分{:02d}秒".format(m, s)
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        out_str = "第{}轮训练完成，耗时{}，\n训练集上的loss={:.6f}；\n验证集上的loss={:.4f}，mIoU={:.6f}\n最好的结果是第{}轮，mIoU={:.6f}" \
            .format(epoch, time_str, train_loss, val_loss, val_miou, best_epoch, best_miou)
        # out_str = "第{}轮训练完成，耗时{}，\n训练集上的segm_loss={:.6f},cls_loss{:.6f}\n验证集上的segm_loss={:.4f},cls_loss={:.4f},mIoU={:.6f}\n最好的结果是第{}轮，mIoU={:.6f}" \
        #     .format(epoch, time_str, train_loss, train_cls_loss, val_loss, val_cls_loss, val_miou, best_epoch,
        #             best_miou)
        print(out_str)
        logger.info(out_str + '\n')
