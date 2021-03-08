from torchvision import transforms as T
import utils.transforms_DL as T_DL

num_classes = 10
input_channel = 4
device = 'cuda:3'
root_dir = '/home/cm/landUseProj/'
logfile = root_dir + f'/code/log/ensemble-0305.log'

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
    T_DL.Normalized_DL(mean=train_mean[:input_channel],
                       std=train_std[:input_channel]),  # 归一化
])

test_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=test_mean[:input_channel], std=test_std[:input_channel]),
])

dataset_cfg = dict(
    train_dir=root_dir + '/tcdata/suichang_round1_train_210120',
    val_dir=root_dir + f'/tcdata/validation0',
    test_dir=root_dir + '/tcdata/suichang_round1_test_partA_210120',
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
    type='EnsembleModel',
    check_point_file_list=[
        root_dir + '/code/checkpoint/smp_unetpp_pretrain_b6_chnl4_rgb_argu_geometry_fold0-0228/smp_unetpp_best.pth',
        root_dir + '/code/checkpoint/smp_unetpp_swa1e5_pretrain_b6_chnl4_rgb_argu_geometry_fold1-0302/smp_unetpp_best.pth',
        root_dir + '/code/checkpoint/smp_unetpp_swa1e5_pretrain_b6_chnl4_rgb_argu_geometry_fold2-0302/smp_unetpp_best.pth',
        root_dir + '/code/checkpoint/smp_unetpp_pretrain_b6_chnl4_rgb_argu_geometry_fold3-0228/smp_unetpp_best.pth',
        root_dir + '/code/checkpoint/smp_unetpp_pretrain_b6_chnl4_rgb_argu_geometry_fold4-0228/smp_unetpp_best.pth',
        #
        # '/home/chiizhang/TC_remote_sense/code/checkpoint/smp_unetpp_crf_pretrain_b7_chnl4-rgb_argu_discolor-alltrain-0221/smp_unetpp_crf_best.pth',
        # # 0.3990
        # '/home/cailinhao/landUseProj_master/landUseProj/code/checkpoint/smp_unetpp_pretrain_b7_chnl4-rgb_argu_discolor-alltrain_100ep-0224/smp_unetpp_best.pth',
        # # 0.3970
        # '/home/cailinhao/landUseProj_master/landUseProj/code/checkpoint/smp_unetpp_atten_pretrain_b7_chnl4-rgb_argu_discolor-alltrain-0225/smp_unetpp_best.pth',
        # # 0.3895
        # '/home/cailinhao/landUseProj_master/landUseProj/code/checkpoint/smp_unetpp_swa1e4_pretrain_b7_chnl4-rgb_argu_discolor-alltrain_100ep-0226/smp_unetpp_best-epoch47.pth',
        # # 0.3969
    ],
    device=device,
    check_point_file=root_dir + f'/code/checkpoint/ensemble-0303/ensemble_best.pth',
)

train_cfg = dict(
    num_workers=6,
    batch_size=64,
    num_epochs=50,
    optimizer_cfg=dict(type='adamw', lr=3e-4, momentum=0.9, weight_decay=5e-4),
    lr_scheduler_cfg=dict(policy='cos', T_0=1, T_mult=1, eta_min=1e-5, last_epoch=-1),
    auto_save_epoch=20,  # 每隔几轮自动保存模型
    is_PSPNet=False,  # 不是PSPNet都设为false
    is_swa=False,
)

test_cfg = dict(
    is_predict=True,  # 是否是预测分类结果
    is_evaluate=False,  # 是否评估模型，也就是计算mIoU
    is_crf=False,
    tta_mode=None,  # 'd4'
    is_ensemble=True,
    # ensemble_weight=[1.0 / 25] * 5 + [1.0 / 5] * 4, # 模型权重，缺省为平均
    dataset='test_dataset',
    batch_size=train_cfg['batch_size'],
    num_workers=train_cfg['num_workers'],
    check_point_file=[
        # '/home/cm/landUseProj/code/checkpoint/smp_unetpp_atte_pretrain_b7_chnl4_rgb_argu_geometry_fold2-0303/smp_unetpp_best.pth',
        '/home/cm/landUseProj/code/checkpoint/ensemble-0303/ensemble_best.pth',

        # root_dir + '/code/checkpoint/smp_unetpp_pretrain_b6_chnl4_rgb_argu_geometry_fold0-0228/smp_unetpp_best.pth',
        # root_dir + '/code/checkpoint/smp_unetpp_swa1e5_pretrain_b6_chnl4_rgb_argu_geometry_fold1-0302/smp_unetpp_best.pth',
        # root_dir + '/code/checkpoint/smp_unetpp_swa1e5_pretrain_b6_chnl4_rgb_argu_geometry_fold2-0302/smp_unetpp_best.pth',
        # root_dir + '/code/checkpoint/smp_unetpp_pretrain_b6_chnl4_rgb_argu_geometry_fold3-0228/smp_unetpp_best.pth',
        # root_dir + '/code/checkpoint/smp_unetpp_pretrain_b6_chnl4_rgb_argu_geometry_fold4-0228/smp_unetpp_best.pth',
        #
        # '/home/chiizhang/TC_remote_sense/code/checkpoint/smp_unetpp_crf_pretrain_b7_chnl4-rgb_argu_discolor-alltrain-0221/smp_unetpp_crf_best.pth',
        # # 0.3990
        # '/home/cailinhao/landUseProj_master/landUseProj/code/checkpoint/smp_unetpp_pretrain_b7_chnl4-rgb_argu_discolor-alltrain_100ep-0224/smp_unetpp_best.pth',
        # # 0.3970
        # '/home/cailinhao/landUseProj_master/landUseProj/code/checkpoint/smp_unetpp_atten_pretrain_b7_chnl4-rgb_argu_discolor-alltrain-0225/smp_unetpp_best.pth',
        # # 0.3895
        # '/home/cailinhao/landUseProj_master/landUseProj/code/checkpoint/smp_unetpp_swa1e4_pretrain_b7_chnl4-rgb_argu_discolor-alltrain_100ep-0226/smp_unetpp_best-epoch47.pth',
        # # 0.3969
    ],
    out_dir=root_dir + '/prediction_result/ensemble_weight-0305/',
)