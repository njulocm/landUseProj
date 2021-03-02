from torchvision import transforms as T
import utils.transforms_DL as T_DL

num_classes = 10
input_channel = 4
device = 'cuda:0'
root_dir = '/home/cailinhao/landUseProj_master/landUseProj/'
logfile = root_dir + '/code/log/smp_unetpp_pretrain_b7_chnl4-rgb_argu_discolor-alltrain_batch16_200ep-0302.log'

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
    # T_DL.RandomColorJitter_DL(p=prob, brightness=1, contrast=1, saturation=1, hue=0.5),  # 概率p调整rgb
    T_DL.Normalized_DL(mean=train_mean[:input_channel],
                       std=train_std[:input_channel]),  # 归一化
])

test_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=test_mean[:input_channel], std=test_std[:input_channel]),
])

dataset_cfg = dict(
    train_dir=root_dir + '/tcdata/suichang_round1_train_210120',
    # train_dir=root_dir + '/tcdata/train',
    val_dir=root_dir + '/tcdata/validation',
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
    type='SMP',
    backbone='efficientnet-b7',
    encoder_weights='imagenet',
    input_channel=input_channel,
    num_classes=num_classes,
    pretrained=True,
    check_point_file=root_dir + '/code/checkpoint/smp_unetpp_pretrain_b7_chnl4-rgb_argu_discolor-alltrain_batch16_200ep-0302/smp_unetpp_best.pth',

    # type='AttUnet',
    # input_channel=4,
    # num_classes=num_classes,
    # check_point_file=root_dir + '/code/checkpoint/AttUnet/AttUnet_model.pth'

    # 使用已训练的模型
    # type='CheckPoint',
    # check_point_file=root_dir + '/code/checkpoint/Unet/Unet_model.pth',
)

train_cfg = dict(
    num_workers=6,
    batch_size=16,
    num_epochs=190,
    optimizer_cfg=dict(type='adamw', lr=3e-4, momentum=0.9, weight_decay=5e-4),
    lr_scheduler_cfg=dict(policy='cos', T_0=6, T_mult=2, eta_min=1e-5),
    # optimizer_cfg=dict(type='sgd', lr=1e-2, momentum=0.9, weight_decay=5e-4),
    # lr_scheduler_cfg=dict(policy='cos', T_0=2, T_mult=2, eta_min=1e-5),
    auto_save_epoch=5,  # 每隔几轮自动保存模型
    is_PSPNet=False,  # 不是PSPNet都设为false
    is_swa=False,
    check_point_file=root_dir + '/code/checkpoint/smp_unetpp_pretrain_b7_chnl4-rgb_argu_discolor-alltrain-0218/smp_unetpp_best.pth',
)

test_cfg = dict(
    is_predict=True,  # 是否是预测分类结果
    is_evaluate=False,  # 是否评估模型，也就是计算mIoU
    is_crf=False,
    tta_mode=None,  # 'd4'
    dataset='test_dataset',
    batch_size=train_cfg['batch_size'],
    num_workers=train_cfg['num_workers'],
    check_point_file=root_dir + '/code/checkpoint/smp_unetpp_pretrain_b7_chnl4-rgb_argu_discolor-alltrain_batch16_200ep-0302/smp_unetpp_best.pth',
    out_dir=root_dir + '/prediction_result/smp_unetpp_pretrain_b7_chnl4-rgb_argu_discolor-alltrain_batch16_200ep-0302/',
)
