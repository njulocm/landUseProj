from torchvision import transforms as T
import utils.transforms_DL as T_DL
import torch

random_seed = 19961002
num_classes = 10
input_channel = 4
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
info = 'SmpUnetpp-b0, alldata train, 几何变换, 色彩增强, batch_size=8, T0=3, 训练45轮'  # 可以在日志开头记录一些补充信息
logfile = f'../user_data/log/round2_b0_SmpUnetpp_color_alldata-0406.log'
# info = 'SmpUnetpp-res18，9:1划分训练和验证集，训练集采用几何变换、无色彩变换，batch_size=8，训练93轮'  # 可以在日志开头记录一些补充信息
# logfile = f'../user_data/log/round2_res18_SmpUnetpp_9v1-0329.log'

train_mean = [0.485, 0.456, 0.406, 0.5]
train_std = [0.229, 0.224, 0.225, 0.25]
test_mean = [0.485, 0.456, 0.406, 0.5]
test_std = [0.229, 0.224, 0.225, 0.25]

# 配置transform
# 注意：train和val涉及到label，需要用带_DL后缀的transform
#      test不涉及label，用原来的transform
prob = 0.5
train_transform = T.Compose([
    T_DL.ToTensor_DL(),  # 转为tensor
    T_DL.RandomFlip_DL(p=prob),  # 概率p水平或者垂直翻转
    T_DL.RandomRotation_DL(p=prob),  # 概率p发生随机旋转(只会90，180，270)
    T_DL.RandomChooseColorJitter_DL(p=prob, brightness=1, contrast=1, saturation=1, hue=0.5),  # 随机选择亮度、对比度、饱和度和色调进行调整
    T_DL.Normalized_DL(mean=train_mean[:input_channel], std=train_std[:input_channel]),  # 归一化
])

val_transform = T.Compose([
    T_DL.ToTensor_DL(),  # 转为tensor
    T_DL.RandomChooseColorJitter_DL(p=prob, brightness=1, contrast=1, saturation=1, hue=0.5),
    T_DL.Normalized_DL(mean=train_mean[:input_channel],
                       std=train_std[:input_channel]),  # 归一化
])

test_transform = T.Compose([
    # T.ToTensor(),
    T.Normalize(mean=test_mean[:input_channel], std=test_std[:input_channel]),
])

dataset_cfg = dict(
    # dir全都改成list
    # train_dir_list=['../tcdata/round2_train/suichang_round1_train_210120',
    #                 '../tcdata/round2_train/suichang_round2_train_210316'],
    train_dir_list=['../tcdata/suichang_round1_train_210120', '../tcdata/suichang_round2_train_210316'],
    split_val_from_train_ratio=None,  # val由train划分而来的比例，如果不采用划分的方式，则置为None，默认由val_dir_list构造验证集
    val_dir_list=['../user_data/round2_val/suichang_round1_train_210120',
                  '../user_data/round2_val/suichang_round2_train_210316'],
    # test_dir_list=['../tcdata/suichang_round1_test_partA_210120'],
    input_channel=input_channel,  # 使用几个通道作为输入
    # 配置transform，三个数据集的配置都要配置
    train_transform=train_transform,
    val_transform=val_transform,
    test_transform=test_transform,
)

model_cfg = dict(
    type='SmpUnetpp',
    backbone='efficientnet-b0',
    # backbone='resnet18',
    decoder_attention_type=None,
    decoder_channels=(256, 128, 64, 32, 16),
    # decoder_attention_type='scse',
    encoder_weights='imagenet',
    input_channel=input_channel,
    num_classes=num_classes,
    pretrained=True,
    device=device,
    check_point_file=f'../user_data/checkpoint/round2_b0_SmpUnetpp_color_alldata-0406/SmpUnetpp_swa_best.pth',
    # check_point_file=f'../user_data/checkpoint/round2_res18_SmpUnetpp_9v1-0329/SmpUnetpp_best.pth',
)

train_cfg = dict(
    num_workers=6,
    batch_size=8,
    num_epochs=48,
    optimizer_cfg=dict(type='adamw', lr=3e-4, momentum=0.9, weight_decay=5e-4),  # 注意学习率调整的倍数
    lr_scheduler_cfg=dict(policy='cos', T_0=1, T_mult=1, eta_min=1e-5, last_epoch=-1),  # swa使用
    auto_save_epoch_list=[23, 47, 71, 95],  # swa需要保存模型的轮数
    is_swa=True,
    check_point_file=f'../user_data/checkpoint/round2_b0_SmpUnetpp_color_alldata-0406/SmpUnetpp_best-epoch44.pth',
    # swa模型读取地址
)

test_cfg = dict(
    test_transform=test_transform,
    processes_num=4,  # 进程数，默认为4，并行推理才会用到
    device='cuda:0',
    # device_available=['cuda:1','cuda:2'],
    boost_type=None,  # None代表加权集成
    is_trt_infer=True,
    FLOAT=32,
    check_point_file=[
        '../user_data/checkpoint/online/SmpUnetpp_swa_best-epoch47.pth',
        '../user_data/checkpoint/round2_b0_SmpUnetpp_color_alldata-0406/SmpUnetpp_swa_best-epoch47.pth',
    ],
)
