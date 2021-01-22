num_classes = 10
device = 'cuda:2'
root_dir = '/home/cm/landUseProj'

dataset_cfg = dict(
    train_dir=root_dir + '/tcdata/suichang_round1_train_210120',
    test_dir=root_dir + '/tcdata/suichang_round1_test_partA_210120',
    input_channel=4,  # 使用几个通道作为输入
    train_ratio=0.8,
    val_ratio=0.2,
    random_seed=999,
)

model_cfg = dict(
    # type='Unet',
    # input_channel=4,
    # num_classes=num_classes,
    # check_point_file=root_dir + '/code/checkpoint/Unet/Unet_model.pth'

    # type='AttUnet',
    # input_channel=4,
    # num_classes=num_classes,
    # check_point_file=root_dir + '/code/checkpoint/AttUnet/AttUnet_model.pth'

    type='NestedUnet',
    input_channel=4,
    num_classes=num_classes,
    check_point_file=root_dir + '/code/checkpoint/NestedUnet/NestedUnet_model.pth'

    # 使用已训练的模型
    # type='CheckPoint',
    # check_point_file=root_dir + '/code/checkpoint/Unet/Unet_model.pth',
)

train_cfg = dict(
    num_workers=4,
    batch_size=8,
    num_epochs=100,
    optimizer_cfg=dict(type="adam", lr=0.01)
)

test_cfg = dict(
    is_predict=False,  # 是否是预测分类结果
    is_evaluate=True,  # 是否评估模型，也就是计算mIoU
    dataset='val_dataset',
    batch_size=train_cfg['batch_size'],
    num_workers=train_cfg['num_workers'],
    check_point_file=root_dir + '/code/checkpoint/Unet/Unet_model-epoch99.pth',
    out_dir=root_dir + '/prediction_result/Unet_val_out-0121',

)
