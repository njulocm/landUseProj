from utils import LandDataset
from torchvision import transforms as T
import utils.transforms_DL as T_DL
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import random
from tqdm import tqdm


def set_seed(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_feature(label, prob, max_num=100):
    # label: 256*256
    # prob: model*10*256*256
    feature_out = None
    land_class_out = []
    for temp_land_class in range(10):
        x_index_list, y_index_list = torch.where(label == temp_land_class)
        index_list = list(range(len(x_index_list)))
        np.random.shuffle(index_list)
        for index in range(min(max_num, len(x_index_list))):
            x = x_index_list[index_list[index]]
            y = y_index_list[index_list[index]]
            temp_feature = torch.unsqueeze(prob[:, :, x, y].flatten(), dim=0)
            if feature_out == None:
                feature_out = temp_feature
            else:
                feature_out = torch.cat((feature_out, temp_feature), dim=0)
            land_class_out.append(temp_land_class)
    # 转为numpy
    feature_out = feature_out.cpu().numpy().astype(np.float16)
    land_class_out = np.array(land_class_out).astype(np.int8)
    return feature_out, land_class_out


def extract_feature_main(dataset, ckpt_files, max_num=100, batch_size=32, device='cuda:0'):
    models = []
    for ckpt in ckpt_files:
        models.append(torch.load(ckpt, map_location=device))

    def _init_fn():
        np.random.seed(6666)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=6,
                            worker_init_fn=_init_fn())

    feature = None
    land_class = None
    write_out_num = 10000  # 从采集到10000个数据开始，每多采集到100000个，就输出一次（覆盖之前结果）
    with torch.no_grad():
        for batch, item in tqdm(enumerate(dataloader)):
            data, label = item
            data = data.to(device)
            out = None
            for model, ckpt in zip(models, ckpt_files):
                temp_out = model(data)
                temp_out = torch.nn.functional.softmax(temp_out, dim=1)  # 转成概率
                temp_out = torch.unsqueeze(temp_out, dim=1)  # batch*model*chnl*w*h
                if out == None:
                    out = temp_out
                else:
                    out = torch.cat((out, temp_out), dim=1)
            # out = out.cpu()  # 取到cpu中
            for i in range(len(out)):
                temp_feature, temp_land_class = extract_feature(label=label[i],
                                                                prob=out[i],
                                                                max_num=max_num)
                if feature is None:
                    feature = temp_feature
                    land_class = temp_land_class
                else:
                    feature = np.concatenate((feature, temp_feature), axis=0)
                    land_class = np.concatenate((land_class, temp_land_class), axis=0)

                # 输出
                if len(feature) > write_out_num:
                    np.save('/home/cm/landUseProj/user_data/feature_sample200.npy', feature)
                    np.save('/home/cm/landUseProj/user_data/land_class_sample200.npy', land_class)
                    write_out_num += 100000

    np.save('/home/cm/landUseProj/user_data/feature_sample200.npy', feature)
    np.save('/home/cm/landUseProj/user_data/land_class_sample200.npy', land_class)
    # np.savetxt('/home/cm/landUseProj/user_data/feature.csv', feature, delimiter=',')
    # np.savetxt('/home/cm/landUseProj/user_data/land_class.csv', land_class, delimiter=',')


if __name__ == '__main__':
    set_seed(6666)
    input_channel = 4
    device = 'cuda:0'
    batch_size = 32
    max_num = 200
    root_dir = '/home/cm/landUseProj/'
    train_dir = root_dir + '/tcdata/suichang_round1_train_210120'
    train_mean = [0.485, 0.456, 0.406, 0.5]
    train_std = [0.229, 0.224, 0.225, 0.25]
    test_mean = [0.485, 0.456, 0.406, 0.5]
    test_std = [0.229, 0.224, 0.225, 0.25]
    train_transform = T.Compose([
        T_DL.ToTensor_DL(),  # 转为tensor
        T_DL.Normalized_DL(mean=train_mean[:input_channel], std=train_std[:input_channel]),  # 归一化
    ])
    train_dataset = LandDataset(DIR=train_dir,
                                mode='train',
                                input_channel=4,
                                transform=train_transform)
    ckpt_files = [
        # b6+swa
        root_dir + '/code/checkpoint/smp_unetpp_swa1e5_pretrain_b6_chnl4_rgb_argu_geometry_fold0-0302/smp_unetpp_best.pth',
        root_dir + '/code/checkpoint/smp_unetpp_swa1e5_pretrain_b6_chnl4_rgb_argu_geometry_fold1-0302/smp_unetpp_best.pth',
        root_dir + '/code/checkpoint/smp_unetpp_swa1e5_pretrain_b6_chnl4_rgb_argu_geometry_fold2-0302/smp_unetpp_best.pth',
        root_dir + '/code/checkpoint/smp_unetpp_swa1e5_pretrain_b6_chnl4_rgb_argu_geometry_fold3-0302/smp_unetpp_best.pth',
        root_dir + '/code/checkpoint/smp_unetpp_swa1e5_pretrain_b6_chnl4_rgb_argu_geometry_fold4-0302/smp_unetpp_best.pth',

        # b7+atte+swa
        '/home/cm/landUseProj/code/checkpoint/smp_unetpp_atte_pretrain_b7_chnl4_rgb_argu_geometry_swa1e5_fold0-0303/smp_unetpp_best.pth',
        '/home/cm/landUseProj/code/checkpoint/smp_unetpp_atte_pretrain_b7_chnl4_rgb_argu_geometry_swa1e5_fold1-0303/smp_unetpp_best.pth',
        '/home/cm/landUseProj/code/checkpoint/smp_unetpp_atte_pretrain_b7_chnl4_rgb_argu_geometry_swa1e5_fold2-0303/smp_unetpp_best.pth',
        '/home/cm/landUseProj/code/checkpoint/smp_unetpp_atte_pretrain_b7_chnl4_rgb_argu_geometry_swa1e5_fold3-0303/smp_unetpp_best.pth',
        '/home/cm/landUseProj/code/checkpoint/smp_unetpp_atte_pretrain_b7_chnl4_rgb_argu_geometry_swa1e5_fold4-0303/smp_unetpp_best.pth',

        # others
        '/home/chiizhang/TC_remote_sense/code/checkpoint/smp_unetpp_crf_pretrain_b7_chnl4-rgb_argu_discolor-alltrain-0221/smp_unetpp_crf_best.pth',
        # 0.3990
        '/home/cailinhao/landUseProj_master/landUseProj/code/checkpoint/smp_unetpp_pretrain_b7_chnl4-rgb_argu_discolor-alltrain_100ep-0224/smp_unetpp_best.pth',
        # 0.3970
        '/home/cailinhao/landUseProj_master/landUseProj/code/checkpoint/smp_unetpp_atten_pretrain_b7_chnl4-rgb_argu_discolor-alltrain-0225/smp_unetpp_best.pth',
        # 0.3895
        '/home/cailinhao/landUseProj_master/landUseProj/code/checkpoint/smp_unetpp_swa1e4_pretrain_b7_chnl4-rgb_argu_discolor-alltrain_100ep-0226/smp_unetpp_best-epoch47.pth',
        # 0.3969
    ]

    extract_feature_main(dataset=train_dataset,
                         ckpt_files=ckpt_files,
                         device=device,
                         batch_size=batch_size,
                         max_num=max_num)
