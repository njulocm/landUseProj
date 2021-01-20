num_classes = 10
device = 'cuda:1'
root_dir = '/home/cm/landUseProj'

dataset_cfg = dict(
    train_dir=root_dir + '/tcdata/suichang_round1_train_210120',
    test_dir=root_dir + '/tcdata/suichang_round1_test_partA_210120',
    input_channel=3,  # 使用几个通道作为输入
    train_ratio=0.8,
    val_ratio=0.2,
    random_seed=999,
)

model_cfg = dict(
    type='FCN',
    num_classes=num_classes,
    show_params=False,
    check_point_file=root_dir + '/code/checkpoint/FCN/fcn_model.pth'
)

train_cfg = dict(
    num_workers=4,
    batch_size=16,
    num_epochs=100,
    optimizer_cfg=dict(type="adam", lr=0.01)
)

test_cfg = dict(
    dataset='val_dataset',
    check_point_file=root_dir + '/code/checkpoint/FCN0/FCN-99.pth',
    out_dir=root_dir + '/prediction_result/fcn_test_out'
)
