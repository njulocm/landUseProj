from torchvision import transforms as T
import utils.transforms_DL as T_DL
import torch
import os

random_seed = 6666
num_classes = 10
input_channel = 4

is_parallel = True  # 多卡训练设为True，单卡训练设为False，或者注释掉
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # 多卡训练，要设置可见的卡，device设为cuda即可，单卡直接注释掉，device正常设置
os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2"
device = 'cuda'
# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

info = 'SmpUnetpp-b7，scse-atte，两张卡训练，对验证集做旋转、翻转和色彩变换増广，batch_size=16，全数据训练90轮'  # 可以在日志开头记录一些补充信息
logfile = f'../user_data/log/round2_parallel_b7_SmpUnetpp_alltrain_valArg-0324.log'

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
    # T_DL.RandomColorJitter_DL(p=prob, brightness=1, contrast=1, saturation=1, hue=0.5),  # 概率p调整rgb
    T_DL.Normalized_DL(mean=train_mean[:input_channel], std=train_std[:input_channel]),  # 归一化
])

val_transform = T.Compose([
    T_DL.ToTensor_DL(),  # 转为tensor
    T_DL.RandomFlip_DL(p=prob),  # 概率p水平或者垂直翻转
    T_DL.RandomRotation_DL(p=prob),  # 概率p发生随机旋转(只会90，180，270)
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
    train_dir_list=['../tcdata/suichang_round1_train_210120', '../tcdata/suichang_round2_train_210316'],
    split_val_from_train_ratio=0.1,  # val由train划分而来的比例，如果不采用划分的方式，则置为None，默认由val_dir_list构造验证集
    # val_dir_list=['../tcdata/suichang_round1_train_210120'],
    # test_dir_list=['../tcdata/suichang_round1_test_partA_210120'],
    test_dir_list=['../tcdata/suichang_round1_test_partB_210120'],
    input_channel=input_channel,  # 使用几个通道作为输入
    train_ratio=0.8,
    val_ratio=0.2,
    random_seed=999,

    # 配置transform，三个数据集的配置都要配置
    train_transform=train_transform,
    val_transform=val_transform,
    test_transform=test_transform,
)

model_cfg = dict(
    type='SmpUnetpp',
    # type='CheckPoint',
    backbone='efficientnet-b7',
    decoder_attention_type='scse',
    encoder_weights='imagenet',
    input_channel=input_channel,
    num_classes=num_classes,
    pretrained=True,
    device=device,
    check_point_file=f'../user_data/checkpoint/round2_parallel_b7_SmpUnetpp_alltrain_valArg-0324/SmpUnetpp_best.pth',
)

train_cfg = dict(
    num_workers=6,
    batch_size=16,
    num_epochs=90,
    optimizer_cfg=dict(type='adamw', lr=3e-4, momentum=0.9, weight_decay=5e-4),
    # lr_scheduler_cfg=dict(policy='cos', T_0=3, T_mult=2, eta_min=1e-5, last_epoch=-1), # 单卡
    lr_scheduler_cfg=dict(policy='cos', T_0=6, T_mult=2, eta_min=1e-5, last_epoch=-1),  # 双卡
    # lr_scheduler_cfg=dict(policy='cos', T_0=96, T_mult=1, eta_min=1e-5, last_epoch=-1),  # swa使用
    # auto_save_epoch_list=[20, 44, 92, 188, 380],  # 需要保存模型的轮数
    auto_save_epoch_list=[17, 41, 89, 185, 377],  # 双卡
    is_PSPNet=False,  # 不是PSPNet都设为false
    is_swa=False,
    # check_point_file=f'../user_data/checkpoint/round2_b7_SmpUnetpp-alltrain-0317/SmpUnetpp_best.pth',
)

test_cfg = dict(
    test_transform=test_transform,
    processes_num=4,  # 进程数，默认为4，并行推理才会用到
    device='cuda:0',
    boost_type=None,  # None代表加权集成
    check_point_file=[
        '../user_data/checkpoint/round2_parallel_b7_SmpUnetpp_alltrain_valArg-0324/SmpUnetpp_best.pth',
        '../user_data/checkpoint/round2_b7_SmpUnetpp_alltrain_swa24-0322/SmpUnetpp_best.pth',
    ],
)
