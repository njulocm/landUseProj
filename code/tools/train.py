from utils import evaluate_model, LandDataset, moving_average, bn_update
from model import build_model

from pytorch_toolbelt import losses as L
from utils.losses import DiceLoss, SoftCrossEntropyLoss

import torch
from torch.utils.data import random_split, DataLoader
from torch.autograd import Variable
from torch import optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np
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
        Y = model(X)

        loss = 0
        for pred in Y:
            loss += loss_func(pred, label)
        loss /= len(Y)
        # loss = loss_func(Y, label)
        loss_list.append(loss.cpu().item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step(epoch + batch / dataloader.__len__())

    return sum(loss_list) / len(loss_list)


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
        # with autocast():
        #     Y = model(X)
        #     loss = loss_func(Y, label)
        #     loss_list.append(loss.cpu().item())
        #     #
        #     # optimizer.zero_grad()
        #     # loss.backward()
        #     # optimizer.step()
        #
        #     scaler.scale(loss).backward()
        #     scaler.step(optimizer)
        #     scaler.update()
        #     optimizer.zero_grad()

        Y = model(X)
        loss = loss_func(Y, label)
        loss_list.append(loss.cpu().item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
    is_parallel = cfg.setdefault(key='is_parallel', default=False)
    device = cfg.device
    is_online_train = cfg.setdefault(key='is_online_train', default=False)

    # 配置logger
    logging.basicConfig(filename=cfg.logfile,
                        filemode='a',
                        level=logging.INFO,
                        format='%(asctime)s\n%(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()

    #
    # 构建数据集
    train_dataset = LandDataset(DIR_list=dataset_cfg.train_dir_list,
                                mode='train',
                                input_channel=dataset_cfg.input_channel,
                                transform=dataset_cfg.train_transform)
    split_val_from_train_ratio = dataset_cfg.setdefault(key='split_val_from_train_ratio', default=None)
    if split_val_from_train_ratio is None:
        val_dataset = LandDataset(DIR_list=dataset_cfg.val_dir_list,
                                  mode='val',
                                  input_channel=dataset_cfg.input_channel,
                                  transform=dataset_cfg.val_transform)
    else:
        val_size = int(len(train_dataset) * split_val_from_train_ratio)
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = random_split(train_dataset,
                                                  [train_size, val_size],
                                                  generator=torch.manual_seed(cfg.random_seed))
        # val_dataset.dataset.transform = dataset_cfg.val_transform # 要配置一下val的transform
        print(f"按照{split_val_from_train_ratio}切分训练集...")

    # 构建dataloader
    def _init_fn():
        np.random.seed(cfg.random_seed)

    train_dataloader = DataLoader(train_dataset, batch_size=train_cfg.batch_size, shuffle=True,
                                  num_workers=train_cfg.num_workers, drop_last=True, worker_init_fn=_init_fn())
    val_dataloader = DataLoader(val_dataset, batch_size=train_cfg.batch_size, num_workers=train_cfg.num_workers,
                                shuffle=False, drop_last=True, worker_init_fn=_init_fn())

    # 构建模型
    if train_cfg.is_swa:
        model = torch.load(train_cfg.check_point_file,
                           map_location=device).to(device)  # device参数传在里面，不然默认是先加载到cuda:0，to之后再加载到相应的device上
        swa_model = torch.load(train_cfg.check_point_file,
                               map_location=device).to(device)  # device参数传在里面，不然默认是先加载到cuda:0，to之后再加载到相应的device上
        if is_parallel:
            model = torch.nn.DataParallel(model)
            swa_model = torch.nn.DataParallel(swa_model)
        swa_n = 0
        parameters = swa_model.parameters()
    else:
        model = build_model(model_cfg).to(device)
        if is_parallel:
            model = torch.nn.DataParallel(model)
        parameters = model.parameters()

    # 定义优化器
    optimizer_cfg = train_cfg.optimizer_cfg
    lr_scheduler_cfg = train_cfg.lr_scheduler_cfg
    if optimizer_cfg.type == 'adam':
        optimizer = optim.Adam(params=parameters,
                               lr=optimizer_cfg.lr,
                               weight_decay=optimizer_cfg.weight_decay)
    elif optimizer_cfg.type == 'adamw':
        optimizer = optim.AdamW(params=parameters,
                                lr=optimizer_cfg.lr, weight_decay=optimizer_cfg.weight_decay)
    elif optimizer_cfg.type == 'sgd':
        optimizer = optim.SGD(params=parameters,
                              lr=optimizer_cfg.lr,
                              momentum=optimizer_cfg.momentum,
                              weight_decay=optimizer_cfg.weight_decay)
    elif optimizer_cfg.type == 'RMS':
        optimizer = optim.RMSprop(params=parameters, lr=optimizer_cfg.lr,
                                  weight_decay=optimizer_cfg.weight_decay)
    else:
        raise Exception('没有该优化器！')

    if not lr_scheduler_cfg:
        lr_scheduler = None
    elif lr_scheduler_cfg.policy == 'cos':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, lr_scheduler_cfg.T_0,
                                                                            lr_scheduler_cfg.T_mult,
                                                                            lr_scheduler_cfg.eta_min,
                                                                            last_epoch=lr_scheduler_cfg.last_epoch)
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
    auto_save_epoch_list = train_cfg.setdefault(key='auto_save_epoch_list', default=5)  # 每隔几轮保存一次模型，默认为5
    train_loss_list = []
    val_loss_list = []
    val_loss_min = 999999
    best_epoch = 0
    best_miou = 0
    train_loss = 10  # 设置一个初始值
    logger.info('开始在{}上训练{}模型...'.format(device, model_cfg.type))
    logger.info('补充信息：{}\n'.format(cfg.setdefault(key='info', default='None')))
    for epoch in range(train_cfg.num_epochs):
        print()
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        start_time = time.time()
        print(f"正在进行第{epoch}轮训练...")
        logger.info('*' * 10 + f"第{epoch}轮" + '*' * 10)
        #
        # 训练一轮
        if train_cfg.is_swa:  # swa训练方式
            train_loss = train_epoch(swa_model, optimizer, lr_scheduler, loss_func, train_dataloader, epoch, device)
            moving_average(model, swa_model, 1.0 / (swa_n + 1))
            swa_n += 1
            bn_update(train_dataloader, model, device)
        else:
            train_loss = train_epoch(model, optimizer, lr_scheduler, loss_func, train_dataloader, epoch, device)
            # train_loss = train_unet3p_epoch(model, optimizer, lr_scheduler, loss_func, train_dataloader, epoch, device)

        #
        # 在训练集上评估模型
        # val_loss, val_miou = evaluate_unet3p_model(model, val_dataset, loss_func, device,
        #                                     cfg.num_classes, train_cfg.num_workers, batch_size=train_cfg.batch_size)
        if not is_online_train:  # 只有在线下训练的时候才需要评估模型
            val_loss, val_miou = evaluate_model(model, val_dataloader, loss_func, device, cfg.num_classes)
        else:
            val_loss = 0
            val_miou = 0

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        # 保存模型
        if not is_online_train: # 非线上训练时需要保存best model
            if val_loss < val_loss_min:
                val_loss_min = val_loss
                best_epoch = epoch
                best_miou = val_miou
                if is_parallel:
                    torch.save(model.module, model_cfg.check_point_file)
                else:
                    torch.save(model, model_cfg.check_point_file)

        if epoch in auto_save_epoch_list:  # 如果再需要保存的轮次中，则保存
            model_file = model_cfg.check_point_file.split('.pth')[0] + '-epoch{}.pth'.format(epoch)
            if is_parallel:
                torch.save(model.module, model_file)
            else:
                torch.save(model, model_file)

        # 打印中间结果
        end_time = time.time()
        run_time = int(end_time - start_time)
        m, s = divmod(run_time, 60)
        time_str = "{:02d}分{:02d}秒".format(m, s)
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        out_str = "第{}轮训练完成，耗时{}，\t训练集上的loss={:.6f}；\t验证集上的loss={:.4f}，mIoU={:.6f}\t最好的结果是第{}轮，mIoU={:.6f}" \
            .format(epoch, time_str, train_loss, val_loss, val_miou, best_epoch, best_miou)
        # out_str = "第{}轮训练完成，耗时{}，\n训练集上的segm_loss={:.6f},cls_loss{:.6f}\n验证集上的segm_loss={:.4f},cls_loss={:.4f},mIoU={:.6f}\n最好的结果是第{}轮，mIoU={:.6f}" \
        #     .format(epoch, time_str, train_loss, train_cls_loss, val_loss, val_cls_loss, val_miou, best_epoch,
        #             best_miou)
        print(out_str)
        logger.info(out_str + '\n')
