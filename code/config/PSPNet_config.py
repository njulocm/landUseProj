from torchvision import transforms as T
from torch import nn

num_classes = 10
device = 'cuda:3'
root_dir = '/home/cm/landUseProj'
logfile = root_dir+'/code/log/PSPNet_aus-log-0131.log'

train_mean = [0.12115830639715953, 0.13374122921202505, 0.10591787170772765, 0.5273172240088813]
train_std = [0.06708223199001057, 0.07782029730399954, 0.06915748947925031, 0.16104953671241798]
test_mean = [0.1070993648354524, 0.10668084722780714, 0.10053905204822813, 0.4039465719983469]
test_std = [0.0659528024287434, 0.07412065904513164, 0.07394464607772513, 0.1716164042414669]

dataset_cfg = dict(
    train_dir = root_dir + '/tcdata/train',
    # train_dir=root_dir + '/tcdata/argument',
    val_dir=root_dir + '/tcdata/validation',
    test_dir=root_dir + '/tcdata/suichang_round1_test_partA_210120',
    input_channel=4,  # 使用几个通道作为输入
    train_ratio=0.8,
    val_ratio=0.2,
    random_seed=999,

    # 配置transform
    # train_set和val_set会使用train_transform；test_set会使用test_transform
    train_transform=T.Compose([T.ToTensor(),
                               T.Normalize(mean=train_mean, std=train_std)]),
    test_transform=T.Compose([T.ToTensor(),
                              T.Normalize(mean=test_mean, std=test_std)]),
)

model_cfg = dict(
    type='PSPNet',
    aux_weight = 0.4, # 辅助loss权重
    layers=50, # resnet的层数
    input_channel=dataset_cfg['input_channel'],
    num_classes=num_classes,
    pretrained=False,

    bins=(1, 2, 3, 6),
    dropout=0.1,
    zoom_factor=8,
    use_ppm=True,

    check_point_file=root_dir + '/code/checkpoint/PSPNet_aux/PSPNet_aux_model.pth',
    device=device, # 加载checkPoint需要指定device，否则会默认加载到cuda:0而显存溢出

)

train_cfg = dict(
    num_workers=4,
    batch_size=16,
    num_epochs=100,
    # optimizer_cfg=dict(type="adam", lr=0.01),
    optimizer_cfg=dict(type="sgd", lr=0.01, momentum=0.9, weight_decay=1e-4),
    is_PSPNet=True
)

test_cfg = dict(
    is_predict=True,  # 是否是预测分类结果
    is_evaluate=False,  # 是否评估模型，也就是计算mIoU
    dataset='test_dataset',
    batch_size=train_cfg['batch_size'],
    num_workers=train_cfg['num_workers'],
    check_point_file=root_dir + '/code/checkpoint/PSPNet_aux/PSPNet_aux_model-epoch99.pth',
    out_dir=root_dir + '/prediction_result/PSPNet_aux_test_out-0130',
)
