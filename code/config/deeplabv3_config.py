from torchvision import transforms as T
import utils.transforms_DL as T_DL

num_classes = 10
input_channel = 3
device = 'cuda:3'
root_dir = '/home/cailinhao/landUseProj_master/landUseProj/'
logfile = root_dir + '/code/log/deeplabv3-argu_color-0202.log'

train_mean = [0.485, 0.456, 0.406, 0.5]
train_std = [0.229, 0.224, 0.225, 0.25]
test_mean = [0.485, 0.456, 0.406, 0.5]
test_std = [0.229, 0.224, 0.225, 0.25]

# 配置transform
# 注意：train和val涉及到label，需要用带_DL后缀的transform
#      test不涉及label，用原来的transform
prob = 0.25
train_transform = T.Compose([
    T_DL.ToTensor_DL(),  # 转为tensor
    T_DL.RandomFlip_DL(p=prob),  # 概率p水平或者垂直翻转
    # T_DL.RandomRotation_DL(p=prob),  # 概率p发生随机旋转(只会90，180，270)
    T_DL.RandomColorJitter_DL(p=prob, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),  # 概率p调整rgb
    T_DL.Normalized_DL(mean=train_mean[:input_channel if input_channel != 4 else -1], std=train_std[:input_channel if input_channel != 4 else -1]),  # 归一化
])

val_transform = T.Compose([
    T_DL.ToTensor_DL(),  # 转为tensor
    T_DL.Normalized_DL(mean=train_mean[:input_channel if input_channel != 4 else -1],
                       std=train_std[:input_channel if input_channel != 4 else -1]),  # 归一化
])

test_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=test_mean[:input_channel if input_channel != 4 else -1], std=test_std[:input_channel if input_channel != 4 else -1]),
])

dataset_cfg = dict(
    # train_dir=root_dir + '/tcdata/suichang_round1_train_210120',
    # train_dir = root_dir + '/tcdata/train'
    train_dir=root_dir + '/tcdata/train',
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
    type='DeepLabV3P',
    input_channel=input_channel,
    num_classes=num_classes,
    backbone='resnet',  # 'resnet', 'xception', 'drn', 'mobilenet'
    out_stride=8,  # 8,16
    sync_bn=False,
    freeze_bn=False,
    pretrained=True,
    check_point_file=root_dir + '/code/checkpoint/deeplabv3-argu_color-0202/deeplabv3_best.pth',

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
    # optimizer_cfg=dict(type="adam", lr=0.01, weight_decay=1e-4),
    optimizer_cfg=dict(type="sgd", lr=0.02, momentum=0.9, weight_decay=0.0005),
    lr_scheduler_cfg=dict(policy='poly', power=0.9, min_lr=1e-4),
    auto_save_epoch=5,  # 每隔几轮自动保存模型
    is_PSPNet=False  # 不是PSPNet都设为false
)

test_cfg = dict(
    is_predict=True,  # 是否是预测分类结果
    is_evaluate=False,  # 是否评估模型，也就是计算mIoU
    dataset='test_dataset',
    batch_size=train_cfg['batch_size'],
    num_workers=train_cfg['num_workers'],
    check_point_file=root_dir + '/code/checkpoint/deeplabv3/deeplabv3_best.pth',
    out_dir=root_dir + '/prediction_result/deeplabv3-0201',
)
