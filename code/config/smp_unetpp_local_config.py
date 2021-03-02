from torchvision import transforms as T
import utils.transforms_DL as T_DL

num_classes = 10
input_channel = 4
random_seed = 6666
device = 'cuda'
root_dir = 'S:/landUseProj/'
logfile = root_dir + '/code/log/local_random.log'

train_mean = [0.485, 0.456, 0.406, 0.5]
train_std = [0.229, 0.224, 0.225, 0.25]
test_mean = [0.485, 0.456, 0.406, 0.5]
test_std = [0.229, 0.224, 0.225, 0.25]

# ����transform
# ע�⣺train��val�漰��label����Ҫ�ô�_DL��׺��transform
#      test���漰label����ԭ����transform
prob = 0.5
train_transform = T.Compose([
    T_DL.ToTensor_DL(),  # תΪtensor
    # T_DL.RandomFlip_DL(p=prob),  # ����pˮƽ���ߴ�ֱ��ת
    # T_DL.RandomRotation_DL(p=prob),  # ����p���������ת(ֻ��90��180��270)
    # T_DL.RandomColorJitter_DL(p=prob, brightness=1, contrast=1, saturation=1, hue=0.5),  # ����p����rgb
    T_DL.Normalized_DL(mean=train_mean[:input_channel], std=train_std[:input_channel]),  # ��һ��
])

val_transform = T.Compose([
    T_DL.ToTensor_DL(),  # תΪtensor
    # T_DL.RandomColorJitter_DL(p=prob, brightness=1, contrast=1, saturation=1, hue=0.5),  # ����p����rgb
    T_DL.Normalized_DL(mean=train_mean[:input_channel],
                       std=train_std[:input_channel]),  # ��һ��
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
    input_channel=input_channel,  # ʹ�ü���ͨ����Ϊ����
    train_ratio=0.8,
    val_ratio=0.2,
    random_seed=999,

    # ����transform���������ݼ������ö�Ҫ����
    train_transform=train_transform,
    val_transform=val_transform,
    test_transform=test_transform,
)

model_cfg = dict(
    type='SMP',
    backbone='efficientnet-b6',
    encoder_weights='imagenet',
    input_channel=input_channel,
    num_classes=num_classes,
    pretrained=True,
    check_point_file=root_dir + '/code/checkpoint/local_random/smp_unetpp_best.pth',

    # type='AttUnet',
    # input_channel=4,
    # num_classes=num_classes,
    # check_point_file=root_dir + '/code/checkpoint/AttUnet/AttUnet_model.pth'

    # ʹ����ѵ����ģ��
    # type='CheckPoint',
    # check_point_file=root_dir + '/code/checkpoint/Unet/Unet_model.pth',
)

train_cfg = dict(
    num_workers=4,
    batch_size=6,
    num_epochs=95,
    optimizer_cfg=dict(type='adamw', lr=3e-4, momentum=0.9, weight_decay=5e-4),
    lr_scheduler_cfg=dict(policy='cos', T_0=3, T_mult=2, eta_min=1e-5),
    # optimizer_cfg=dict(type='sgd', lr=1e-2, momentum=0.9, weight_decay=5e-4),
    # lr_scheduler_cfg=dict(policy='cos', T_0=2, T_mult=2, eta_min=1e-5),
    auto_save_epoch=5,  # ÿ�������Զ�����ģ��
    is_PSPNet=False,  # ����PSPNet����Ϊfalse
    is_swa=False,
    check_point_file=root_dir + '/code/checkpoint/smp_unetpp_pretrain_b7_chnl4-rgb_argu_discolor-alltrain-0218/smp_unetpp_best.pth',
)

test_cfg = dict(
    is_predict=True,  # �Ƿ���Ԥ�������
    is_evaluate=False,  # �Ƿ�����ģ�ͣ�Ҳ���Ǽ���mIoU
    is_crf=False,
    tta_mode=None,  # 'd4'
    dataset='test_dataset',
    batch_size=train_cfg['batch_size'],
    num_workers=train_cfg['num_workers'],
    check_point_file=root_dir + '/code/checkpoint/local_random/smp_unetpp_best.pth',
    out_dir=root_dir + '/prediction_result/local_random/',
)
