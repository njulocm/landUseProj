import numpy as np
import torch

def adjust_learning_rate(lr_scheduler_cfg, optimizer, progress_rate, lr_init):
    if lr_scheduler_cfg.policy == 'poly':
        lr = lr_init * pow((1 - 1.0 * progress_rate), lr_scheduler_cfg.power)
        # power用来控制学习率曲线的形状, power<1, 曲线凸起来,下降慢后快 ;power>1, 凹下去, 下降先快后慢
        if lr < lr_scheduler_cfg.min_lr:
            lr = lr_scheduler_cfg.min_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr