from torchvision import transforms as T
import utils.transforms_DL as T_DL
import torch

random_seed = 6666
num_classes = 10
input_channel = 4
device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
info = 'SmpUnetpp-b0, alldata train, train_set use geometry transform, without color transform, batch_size=8, T0=2, train 126 epochs'  # 可以在日志开头记录一些补充信息
logfile = f'../user_data/log/round2_b0_SmpUnetpp_T2-0404.log'
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
    # T_DL.RandomChooseColorJitter_DL(p=prob, brightness=1, contrast=1, saturation=1, hue=0.5),  # 随机选择亮度、对比度、饱和度和色调进行调整
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
    val_dir_list=['../tcdata/round2_val/suichang_round1_train_210120',
                  '../tcdata/round2_val/suichang_round2_train_210316'],
    # test_dir_list=['../tcdata/suichang_round1_test_partA_210120'],
    input_channel=input_channel,  # 使用几个通道作为输入
    train_ratio=0.8,
    val_ratio=0.2,
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
    check_point_file=f'../user_data/checkpoint/round2_b0_SmpUnetpp_T2-0404/SmpUnetpp_best.pth',
    # check_point_file=f'../user_data/checkpoint/round2_res18_SmpUnetpp_9v1-0329/SmpUnetpp_best.pth',
)

train_cfg = dict(
    num_workers=6,
    batch_size=8,
    num_epochs=126,
    optimizer_cfg=dict(type='adamw', lr=3e-4, momentum=0.9, weight_decay=5e-4),  # 注意学习率调整的倍数
    # lr_scheduler_cfg=dict(policy='cos', T_0=3, T_mult=2, eta_min=1e-5, last_epoch=-1),
    # auto_save_epoch_list=[20, 44, 92, 188, 380],  # 需要保存模型的轮数, T0=3
    # lr_scheduler_cfg=dict(policy='cos', T_0=1, T_mult=1, eta_min=1e-5, last_epoch=-1),  # swa使用
    # auto_save_epoch_list=[11, 23, 35, 47],  # swa需要保存模型的轮数
    lr_scheduler_cfg=dict(policy='cos', T_0=2, T_mult=2, eta_min=1e-5, last_epoch=-1),
    auto_save_epoch_list=[29, 61, 125, 253],  # 需要保存模型的轮数, T0=2
    is_PSPNet=False,  # 不是PSPNet都设为false
    is_swa=False,
    # check_point_file=f'../user_data/checkpoint/round2_b0_SmpUnetpp_swa48_alldata-0330/SmpUnetpp_best-epoch47.pth',
    # swa模型读取地址
)

test_cfg = dict(
    test_transform=test_transform,
    processes_num=4,  # 进程数，默认为4，并行推理才会用到
    device='cuda:0',
    # device_available=['cuda:1','cuda:2'],
    boost_type=None,  # None代表加权集成
    check_point_file=[
        # '../user_data/checkpoint/round2_b0_SmpUnetpp_swa48*2-0402/SmpUnetpp_best-epoch47.pth',
        '../user_data/checkpoint/round2_b0_SmpUnetpp_swa48_alldata-0330/SmpUnetpp_best-epoch47.pth',
        # '../user_data/checkpoint/round2_b0_depth4_256_SmpUnetpp-0330/SmpUnetpp_best.pth',
        # '../user_data/checkpoint/round2_b0_depth4_256_SmpUnetpp_swa48-0331/SmpUnetpp_best-epoch47.pth',
        # '../user_data/checkpoint/round2_b0_depth4_128_SmpUnetpp-0330/SmpUnetpp_best-epoch44.pth'
        # '../user_data/checkpoint/round2_b0_SmpUnet_9v1-0329/SmpUnetpp_best-epoch44.pth',
        # '../user_data/checkpoint/round2_b0_SmpUnetpp_color_9v1-0329/SmpUnetpp_best-epoch44.pth',
        # '../user_data/checkpoint/round2_b0_SmpUnetpp_9v1-0327/SmpUnetpp_best-epoch92.pth',
        # '../user_data/checkpoint/round2_b0_SmpUnetpp_9v1-0327/SmpUnetpp_best.pth',
        # '../user_data/checkpoint/round2_b0_SmpUnetpp_batch32_9v1-0329/SmpUnetpp_best-epoch44.pth',
        # '../user_data/checkpoint/round2_b0_SmpUnetpp_atte_alldata-0329/SmpUnetpp_best-epoch44.pth',

        # '../user_data/checkpoint/round2_res18_SmpUnetpp_9v1-0329/SmpUnetpp_best-epoch44.pth',

        # '../user_data/checkpoint/round2_parallel_b7_SmpUnetpp_alltrain_valArg-0324/SmpUnetpp_best.pth',
        # '../user_data/checkpoint/round2/round2_b7_SmpUnetpp_alltrain_swa24-0322/SmpUnetpp_best.pth',
        # '../user_data/checkpoint/round2_unet_color_9v1-0326/Unet_best-epoch44.pth',
        # '../user_data/checkpoint/round2_regx320_SmpUnetpp-alltrain-0323/SmpUnetpp_best-epoch44.pth',
        # '../user_data/checkpoint/round2_parallel_b7_SmpUnetpp-alltrain-0322/SmpUnetpp_best-epoch41.pth',
        # '../user_data/checkpoint/round2_parallel_b7_SmpUnetpp_alltrain_valArg-0324/SmpUnetpp_best-epoch41.pth',

        # '../user_data/checkpoint/round1/smp_unetpp_pretrain_b7_chnl4-rgb_argu_discolor-alltrain_100ep-0224/smp_unetpp_best.pth',
    ],
    is_trt_infer=True,
    FLOAT=32, # 好像没有起作用，不知道是不是命令行的原因
    trt_file='../user_data/checkpoint/round2_b0_SmpUnetpp_swa48_alldata-0330/SmpUnetpp_best-epoch47.trt',
)

# test_cfg = dict(
#     is_predict=True,  # 是否是预测分类结果
#     is_evaluate=True,  # 是否评估模型，也就是计算mIoU
#     test_transform=test_transform,
#     is_crf=False,
#     tta_mode=None,  # 'd4'
#     is_ensemble=True,
#     # processes_num=4, # 进程数，默认为4，并行推理才会用到
#     device='cuda:0',
#     # device_available=['cuda:3'],  # 可用的推理设备，默认只用'cuda:0'
#     # ensemble_weight=[0.4 / 5] * 5 + [0.6 / 4] * 4,  # 模型权重，缺省为平均
#     # ensemble_weight=[0.2 / 5] * 5 + [0.2 / 5] * 5 + [0.6 / 4] * 4,  # 模型权重，缺省为平均
#     boost_type=None,  # None代表加权集成
#     # boost_ckpt_file='/home/cm/landUseProj/code/checkpoint/adaBoost/adaBoost_b6_b7_others.pkl',
#     # boost_ckpt_file='/home/cm/landUseProj/code/checkpoint/adaBoost/xgBoost_b6_b7_other_sample200_iter1000.pkl',
#     dataset='val_dataset',
#     batch_size=8,
#     num_workers=train_cfg['num_workers'],
#     processes_num=4,
#     is_model_half=False,  # 是否采用半精度推理
#     check_point_file=[
#         # '../user_data/checkpoint/round2_unet_color_9v1-0326/Unet_best-epoch44.pth',
#         # '../user_data/checkpoint/round2_regx320_SmpUnetpp-alltrain-0323/SmpUnetpp_best-epoch44.pth',
#         # '../user_data/checkpoint/round2_b7_SmpUnetpp_alltrain_swa24-0322/SmpUnetpp_best.pth',
#         # '../user_data/checkpoint/round2_parallel_b7_SmpUnetpp-alltrain-0322/SmpUnetpp_best-epoch41.pth',
#         # '../user_data/checkpoint/round2_parallel_b7_SmpUnetpp_alltrain_valArg-0324/SmpUnetpp_best-epoch41.pth',
#
#         # '../user_data/checkpoint/round2_b7_SmpUnetpp-alltrain-0317/SmpUnetpp_best-epoch44.pth',
#         # '../user_data/checkpoint/round2_parallel_b7_SmpUnetpp-alltrain-0318/SmpUnetpp_best.pth',
#         # '../user_data/checkpoint/round1/smp_unetpp_pretrain_b7_chnl4-rgb_argu_discolor-alltrain_100ep-0224/smp_unetpp_best.pth',
#
#         # b6+swa
#         # '../user_data/checkpoint/round1/smp_unetpp_swa1e5_pretrain_b6_chnl4_rgb_argu_geometry_fold0-0302/smp_unetpp_best.pth',
#         # '../user_data/checkpoint/round1/smp_unetpp_swa1e5_pretrain_b6_chnl4_rgb_argu_geometry_fold1-0302/smp_unetpp_best.pth',
#         # '../user_data/checkpoint/round1/smp_unetpp_swa1e5_pretrain_b6_chnl4_rgb_argu_geometry_fold2-0302/smp_unetpp_best.pth',
#         # '../user_data/checkpoint/round1/smp_unetpp_swa1e5_pretrain_b6_chnl4_rgb_argu_geometry_fold3-0302/smp_unetpp_best.pth',
#         # '../user_data/checkpoint/round1/smp_unetpp_swa1e5_pretrain_b6_chnl4_rgb_argu_geometry_fold4-0302/smp_unetpp_best.pth',
#
#         # b7+atte+swa
#         # '../user_data/checkpoint/round1/smp_unetpp_atte_pretrain_b7_chnl4_rgb_argu_geometry_swa1e5_fold0-0303/smp_unetpp_best.pth',
#         # '../user_data/checkpoint/round1/smp_unetpp_atte_pretrain_b7_chnl4_rgb_argu_geometry_swa1e5_fold1-0303/smp_unetpp_best.pth',
#         # '../user_data/checkpoint/round1/smp_unetpp_atte_pretrain_b7_chnl4_rgb_argu_geometry_swa1e5_fold2-0303/smp_unetpp_best.pth',
#         # '../user_data/checkpoint/round1/smp_unetpp_atte_pretrain_b7_chnl4_rgb_argu_geometry_swa1e5_fold3-0303/smp_unetpp_best.pth',
#         # '../user_data/checkpoint/round1/smp_unetpp_atte_pretrain_b7_chnl4_rgb_argu_geometry_swa1e5_fold4-0303/smp_unetpp_best.pth',
#
#         # others
#         # '../user_data/checkpoint/round1/smp_unetpp_crf_pretrain_b7_chnl4-rgb_argu_discolor-alltrain-0221/smp_unetpp_crf_best.pth',
#         # # 0.3985
#         # '../user_data/checkpoint/round1/smp_unetpp_pretrain_b7_chnl4-rgb_argu_discolor-alltrain_100ep-0224/smp_unetpp_best.pth',
#         # # 0.3970
#         # '../user_data/checkpoint/round1/smp_unetpp_atten_pretrain_b7_chnl4-rgb_argu_discolor-alltrain-0225/smp_unetpp_best.pth',
#         # # 0.3895
#         # '../user_data/checkpoint/round1/smp_unetpp_swa1e4_pretrain_b7_chnl4-rgb_argu_discolor-alltrain_100ep-0226/smp_unetpp_best-epoch47.pth',
#         # # 0.3969
#     ],
#     out_dir='../prediction_result/',
# )
