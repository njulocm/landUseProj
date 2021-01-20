from torch.utils.data.dataloader import DataLoader
from utils import evaluate_model, LandDataset
from model import build_model
import torch
from torch import nn
from torch.utils.data import random_split, DataLoader
from torch import optim
import time


def train_epoch(model, optimizer, loss_func, dataloader, device):
    '''
    :param model:
    :param optimizer:
    :param loss_func:
    :param dataloader:
    :param device:
    :return: 返回该轮训练的平均loss
    '''
    loss_list = []
    for batch, item in enumerate(dataloader):
        # print(f"batch={batch}")
        X, label = item
        X = X.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        Y = model(X)
        loss = loss_func(Y, label)
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

    #
    # 构建数据集
    land_dataset = LandDataset(dataset_cfg.train_dir, input_channel=dataset_cfg.input_channel)
    # 划分数据集
    train_size = int(dataset_cfg.train_ratio * len(land_dataset))
    val_size = len(land_dataset) - train_size
    train_dataset, val_dataset = random_split(land_dataset, [train_size, val_size],
                                              generator=torch.manual_seed(dataset_cfg.random_seed))

    # 构建dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=train_cfg.batch_size, shuffle=True,
                                  num_workers=train_cfg.num_workers)

    # 构建模型
    model = build_model(model_cfg).to(device)

    # 定义优化器
    optimizer_cfg = train_cfg.optimizer_cfg
    if optimizer_cfg.type == 'adam':
        optimizer = optim.Adam(params=model.parameters(), lr=optimizer_cfg.lr)
    elif optimizer_cfg.type == 'sgd':
        pass  # 待补充
    else:
        raise Exception('没有该优化器！')

    # 定义损失函数
    loss_func = nn.CrossEntropyLoss().to(device)

    # 开始训练
    train_loss_list = []
    val_loss_list = []
    val_loss_min = 999999
    best_epoch = 0
    best_miou = 0
    train_loss = 10  # 设置一个初始值
    for epoch in range(train_cfg.num_epochs):
        print()
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        start_time = time.time()
        print(f"正在进行第{epoch}轮训练...")
        #
        # 训练一轮
        train_loss = train_epoch(model, optimizer, loss_func, train_dataloader, device)

        #
        # 在训练集上评估模型
        val_loss, val_miou = evaluate_model(model, val_dataset, loss_func, device,
                                            cfg.num_classes, train_cfg.num_workers, batch_size=train_cfg.batch_size)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        # 保存模型
        if val_loss < val_loss_min:
            val_loss_min = val_loss
            best_epoch = epoch
            best_miou = val_miou
            torch.save(model, model_cfg.check_point_file)

        if epoch % 10 == 0: # 每10轮保存一次
            model_file = model_cfg.check_point_file.split('.')[0] + '-' + epoch + '.pth'
            torch.save(model, model_file)

        # 打印中间结果
        end_time = time.time()
        run_time = int(end_time - start_time)
        m, s = divmod(run_time, 60)
        time_str = "{:02d}分{:02d}秒".format(m, s)
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print("第{}轮训练完成，耗时{}，\n训练集上的loss={:.6f}；\n验证集上的loss={:.4f}，mIoU={:.6f}\n最好的结果是第{}轮，mIoU={:.6f}" \
              .format(epoch, time_str, train_loss, val_loss, val_miou, best_epoch, best_miou))
