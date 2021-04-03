from torchvision import transforms as T
import utils.transforms_DL as T_DL
import torch

random_seed = 6666
num_classes = 10
input_channel = 4
device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
info = 'Unet, 9:1划分训练和验证集，训练集采用几何变换、无色彩变换，batch_size=16，训练93轮'  # 可以在日志开头记录一些补充信息
logfile = f'../user_data/log/round2_unet_color_9v1-0327.log'

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
    # T_DL.RandomChooseColorJitter_DL(p=prob, brightness=1, contrast=1, saturation=1, hue=0.5),
    # T_DL.RandomColorJitter_DL(p=prob, brightness=1, contrast=1, saturation=1, hue=0.5),  # 概率p调整rgb
    T_DL.Normalized_DL(mean=train_mean[:input_channel], std=train_std[:input_channel]),  # 归一化
])

val_transform = T.Compose([
    T_DL.ToTensor_DL(),  # 转为tensor
    T_DL.Normalized_DL(mean=train_mean[:input_channel],
                       std=train_std[:input_channel]),  # 归一化
])

test_transform = T.Compose([
    # T.ToTensor(),
    T.Normalize(mean=test_mean[:input_channel], std=test_std[:input_channel]),
])

dataset_cfg = dict(
    # dir全都改成list
    train_dir_list=['../tcdata/round2_train/suichang_round1_train_210120',
                    '../tcdata/round2_train/suichang_round2_train_210316'],
    split_val_from_train_ratio=None,  # val由train划分而来的比例，如果不采用划分的方式，则置为None，默认由val_dir_list构造验证集
    val_dir_list=['../tcdata/round2_val/suichang_round1_train_210120',
                  '../tcdata/round2_val/suichang_round2_train_210316'],
    # test_dir_list=['../tcdata/suichang_round1_test_partA_210120'],
    test_dir_list=['../tcdata/suichang_round1_test_partB_210120'],
    input_channel=input_channel,  # 使用几个通道作为输入
    train_ratio=0.8,
    val_ratio=0.2,
    # 配置transform，三个数据集的配置都要配置
    train_transform=train_transform,
    val_transform=val_transform,
    test_transform=test_transform,
)

model_cfg = dict(
    type='Unet',
    # type='CheckPoint',
    input_channel=input_channel,
    num_classes=num_classes,
    device=device,
    check_point_file=f'../user_data/checkpoint/round2_unet_color_9v1-0327/Unet_best.pth',
)

train_cfg = dict(
    num_workers=6,
    batch_size=16,
    num_epochs=93,
    optimizer_cfg=dict(type='adamw', lr=3e-4, momentum=0.9, weight_decay=5e-4),
    lr_scheduler_cfg=dict(policy='cos', T_0=3, T_mult=2, eta_min=1e-5, last_epoch=-1),
    # lr_scheduler_cfg=dict(policy='cos', T_0=1, T_mult=1, eta_min=1e-5, last_epoch=-1),  # swa使用
    auto_save_epoch_list=[20, 44, 92, 188, 380],  # 需要保存模型的轮数
    # auto_save_epoch_list=[], # swa需要保存模型的轮数
    is_PSPNet=False,  # 不是PSPNet都设为false
    is_swa=False,
    # check_point_file=f'../user_data/checkpoint/round2_b7_SmpUnetpp_alltrain_swa24-0322/SmpUnetpp_best.pth',
)

test_cfg = dict(
    is_predict=True,  # 是否是预测分类结果
    is_evaluate=True,  # 是否评估模型，也就是计算mIoU
    test_transform=test_transform,
    is_crf=False,
    tta_mode=None,  # 'd4'
    is_ensemble=True,
    # processes_num=4, # 进程数，默认为4，并行推理才会用到
    device_available=['cuda:0'],  # 可用的推理设备，默认只用'cuda:0'
    # ensemble_weight=[0.4 / 5] * 5 + [0.6 / 4] * 4,  # 模型权重，缺省为平均
    # ensemble_weight=[0.2 / 5] * 5 + [0.2 / 5] * 5 + [0.6 / 4] * 4,  # 模型权重，缺省为平均
    boost_type=None,  # None代表加权集成
    # boost_ckpt_file='/home/cm/landUseProj/code/checkpoint/adaBoost/adaBoost_b6_b7_others.pkl',
    # boost_ckpt_file='/home/cm/landUseProj/code/checkpoint/adaBoost/xgBoost_b6_b7_other_sample200_iter1000.pkl',
    dataset='val_dataset',
    batch_size=8,
    num_workers=train_cfg['num_workers'],
    check_point_file=[
        '../user_data/checkpoint/round2_b7_SmpUnetpp_alltrain_swa24-0322/SmpUnetpp_best.pth',

        # '../user_data/checkpoint/round2_b7_SmpUnetpp-alltrain-0317/SmpUnetpp_best-epoch44.pth',
        # '../user_data/checkpoint/round2_parallel_b7_SmpUnetpp-alltrain-0318/SmpUnetpp_best.pth',
        # '../user_data/checkpoint/round1/smp_unetpp_pretrain_b7_chnl4-rgb_argu_discolor-alltrain_100ep-0224/smp_unetpp_best.pth',

        # b6+swa
        # '../user_data/checkpoint/round1/smp_unetpp_swa1e5_pretrain_b6_chnl4_rgb_argu_geometry_fold0-0302/smp_unetpp_best.pth',
        # '../user_data/checkpoint/round1/smp_unetpp_swa1e5_pretrain_b6_chnl4_rgb_argu_geometry_fold1-0302/smp_unetpp_best.pth',
        # '../user_data/checkpoint/round1/smp_unetpp_swa1e5_pretrain_b6_chnl4_rgb_argu_geometry_fold2-0302/smp_unetpp_best.pth',
        # '../user_data/checkpoint/round1/smp_unetpp_swa1e5_pretrain_b6_chnl4_rgb_argu_geometry_fold3-0302/smp_unetpp_best.pth',
        # '../user_data/checkpoint/round1/smp_unetpp_swa1e5_pretrain_b6_chnl4_rgb_argu_geometry_fold4-0302/smp_unetpp_best.pth',

        # b7+atte+swa
        # '../user_data/checkpoint/round1/smp_unetpp_atte_pretrain_b7_chnl4_rgb_argu_geometry_swa1e5_fold0-0303/smp_unetpp_best.pth',
        # '../user_data/checkpoint/round1/smp_unetpp_atte_pretrain_b7_chnl4_rgb_argu_geometry_swa1e5_fold1-0303/smp_unetpp_best.pth',
        # '../user_data/checkpoint/round1/smp_unetpp_atte_pretrain_b7_chnl4_rgb_argu_geometry_swa1e5_fold2-0303/smp_unetpp_best.pth',
        # '../user_data/checkpoint/round1/smp_unetpp_atte_pretrain_b7_chnl4_rgb_argu_geometry_swa1e5_fold3-0303/smp_unetpp_best.pth',
        # '../user_data/checkpoint/round1/smp_unetpp_atte_pretrain_b7_chnl4_rgb_argu_geometry_swa1e5_fold4-0303/smp_unetpp_best.pth',

        # others
        # '../user_data/checkpoint/round1/smp_unetpp_crf_pretrain_b7_chnl4-rgb_argu_discolor-alltrain-0221/smp_unetpp_crf_best.pth',
        # # 0.3985
        # '../user_data/checkpoint/round1/smp_unetpp_pretrain_b7_chnl4-rgb_argu_discolor-alltrain_100ep-0224/smp_unetpp_best.pth',
        # # 0.3970
        # '../user_data/checkpoint/round1/smp_unetpp_atten_pretrain_b7_chnl4-rgb_argu_discolor-alltrain-0225/smp_unetpp_best.pth',
        # # 0.3895
        # '../user_data/checkpoint/round1/smp_unetpp_swa1e4_pretrain_b7_chnl4-rgb_argu_discolor-alltrain_100ep-0226/smp_unetpp_best-epoch47.pth',
        # # 0.3969
    ],
    out_dir='../prediction_result/',
)
