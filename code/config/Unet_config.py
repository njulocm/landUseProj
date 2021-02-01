from torchvision import transforms as T
import utils.transforms_DL as T_DL

num_classes = 10
device = 'cuda:2'
root_dir = '/home/cm/landUseProj'
logfile = root_dir + '/code/log/Unet_argument_new-0201.log'

train_mean = [0.12115830639715953, 0.13374122921202505, 0.10591787170772765, 0.5273172240088813]
train_std = [0.06708223199001057, 0.07782029730399954, 0.06915748947925031, 0.16104953671241798]
test_mean = [0.1070993648354524, 0.10668084722780714, 0.10053905204822813, 0.4039465719983469]
test_std = [0.0659528024287434, 0.07412065904513164, 0.07394464607772513, 0.1716164042414669]

# 配置transform
# 注意：train和val涉及到label，需要用带_DL后缀的transform
#      test不涉及label，用原来的transform
prob = 0.25
train_transform = T.Compose([
    T_DL.ToTensor_DL(),  # 转为tensor
    T_DL.RandomFlip_DL(p=prob),  # 概率p水平或者垂直翻转
    T_DL.RandomRotation_DL(p=prob),  # 概率p发生随机旋转(只会90，180，270)
    T_DL.RandomColorJitter_DL(p=prob, brightness=1, contrast=1, saturation=1, hue=0.5),  # 概率p调整rgb
    # T_DL.Normalize(mean=train_mean, std=train_std), # 归一化
])

val_transform = T.Compose([
    T_DL.ToTensor_DL(),  # 转为tensor
    # T_DL.Normalize(mean=train_mean, std=train_std), # 归一化
])

test_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=test_mean, std=test_std),
])

dataset_cfg = dict(
    # train_dir=root_dir + '/tcdata/suichang_round1_train_210120',
    # train_dir = root_dir + '/tcdata/train'
    train_dir=root_dir + '/tcdata/train',
    val_dir=root_dir + '/tcdata/validation',
    test_dir=root_dir + '/tcdata/suichang_round1_test_partA_210120',
    input_channel=4,  # 使用几个通道作为输入
    train_ratio=0.8,
    val_ratio=0.2,
    random_seed=999,

    # 配置transform，三个数据集的配置都要配置
    train_transform=train_transform,
    val_transform=val_transform,
    test_transform=test_transform,
)

model_cfg = dict(
    type='Unet',
    input_channel=4,
    num_classes=num_classes,
    check_point_file=root_dir + '/code/checkpoint/Unet_argument_new/Unet_argument_new_model.pth',

    # type='AttUnet',
    # input_channel=4,
    # num_classes=num_classes,
    # check_point_file=root_dir + '/code/checkpoint/AttUnet/AttUnet_model.pth'

    # 使用已训练的模型
    # type='CheckPoint',
    # check_point_file=root_dir + '/code/checkpoint/Unet/Unet_model.pth',
)

train_cfg = dict(
    num_workers=4,
    batch_size=16,
    num_epochs=100,
    optimizer_cfg=dict(type="adam", lr=0.01, weight_decay=1e-4),
    # optimizer_cfg=dict(type="sgd", lr=0.01, momentum=0.9,weight_decay=0.0001),
    auto_save_epoch=2,  # 每隔几轮自动保存模型
    is_PSPNet=False  # 不是PSPNet都设为false
)

test_cfg = dict(
    is_predict=True,  # 是否是预测分类结果
    is_evaluate=False,  # 是否评估模型，也就是计算mIoU
    dataset='test_dataset',
    batch_size=train_cfg['batch_size'],
    # batch_size=4,
    num_workers=train_cfg['num_workers'],
    check_point_file=root_dir + '/code/checkpoint/Unet_argument_new/Unet_argument_model-epoch34.pth',
    out_dir=root_dir + '/prediction_result/Unet_argument_new_test_out-0201',
)
