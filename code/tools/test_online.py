from ai_hub import inferServer
import json
from PIL import Image
from io import BytesIO
import os, random
from torch.autograd import Variable as V
import base64
import requests
import tqdm

import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
import cv2
import joblib
import xgboost
from addict import Dict

from model import build_model

import _thread
import threading
# from multiprocessing import Pool  # 需要针对 IO 密集型任务和 CPU 密集型任务来选择不同的库
from multiprocessing.dummy import Pool  # 需要针对 IO 密集型任务和 CPU 密集型任务来选择不同的库
import time


# from ptflops import get_model_complexity_info


def set_seed(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def test_online_main(cfg):
    infer_cfg = set_infer_cfg(cfg)
    infer = OnlineInfer(infer_cfg)
    infer.run(debuge=False)  # 默认为("127.0.0.1", 80)，可自定义端口，如用于天池大赛请默认即可，指定debuge=True可获得更多报错信息

    # flops, params = get_model_complexity_info(infer.models[0], (4, 224, 224), as_strings=True,print_per_layer_stat=True)
    # print('Flops:  ' + flops)
    # print('Params: ' + params)

    # debug时使用
    # set_seed(6666)
    # start = time.time()
    # for i in range(100):
    #     img = torch.randint(0,255,(256, 256, 4)).to(infer.device)
    #     # img = torch.randn((256, 256, 4), dtype=torch.float16).cuda()
    #     # img = Image.open('/home/cm/landUseProj/tcdata/suichang_round2_train_210316/000001.tif')
    #     # img = np.array(img, dtype=np.uint8)
    #     ret = infer.predict(img)
    #     # ret = infer.predict_parallel(img)
    # run_time = time.time() - start
    # print("耗时:{}秒".format(run_time))


def set_infer_cfg(cfg):
    # config
    test_cfg = cfg.test_cfg
    test_transform = test_cfg.test_transform
    boost_type = test_cfg.setdefault(key='boost_type', default=None)
    ensemble_weight = None
    boost_model = None
    processes_num = test_cfg.setdefault(key='processes_num', default=4)
    pool = Pool(processes=processes_num)  # 进程数可能需要调节
    device = test_cfg.setdefault(key='device', default='cuda:0')
    # device_available = test_cfg.setdefault(key='device_available', default=['cuda:0'])
    is_model_half = test_cfg.setdefault(key='is_model_half', default=False)

    # 加载多个模型
    models = []
    device_list = []  # 需要考虑并行的方法
    models_num = 0
    for ckpt in test_cfg.check_point_file:
        # device = device_available[models_num % len(device_available)]
        model = torch.load(ckpt, map_location=device).half()
        # model_cfg = cfg.model_cfg
        # model = build_model(model_cfg).to(device)
        if isinstance(model, torch.nn.DataParallel):  # 如果是双卡模型，需要去除module
            model = model.module
        models.append(model)
        device_list.append(device)
        models_num += 1

    if boost_type is None:  # 采用加权平均集成
        # 获取模型集成的权重
        ensemble_weight = test_cfg.setdefault(key='ensemble_weight', default=[1.0 / len(models)] * len(models))
        if len(ensemble_weight) != len(models):
            raise Exception('权重个数错误！')
    else:  # 采用boost集成
        if boost_type == 'adaBoost':  # 采用adaBoost集成
            boost_model = joblib.load(test_cfg.boost_ckpt_file)
        elif boost_type == 'XGBoost':  # 采用XGBoost集成
            boost_model = xgboost.Booster(model_file=test_cfg.boost_ckpt_file)

    # 构建model_dict
    infer_cfg = Dict(
        {'models': models,
         'models_num': models_num,
         # 'device_list': device_list,
         'device': device,
         'test_transform': test_transform,
         'boost_type': boost_type,  # 加权为None
         'ensemble_weight': ensemble_weight,
         'boost_model': boost_model,
         'pool': pool,  # 进程池
         'is_model_half': is_model_half,  # 是否采用半精度推理 float16
         }
    )
    return infer_cfg


class OnlineInfer(inferServer):
    def __init__(self, infer_cfg, model=None):
        super().__init__(model)
        # self.infer_cfg = infer_cfg
        self.data = None
        self.models = infer_cfg.models
        self.models_num = infer_cfg.models_num
        # self.device_list = infer_cfg.device_list
        self.device = infer_cfg.device
        self.test_transform = infer_cfg.test_transform
        self.boost_type = infer_cfg.boost_type  # 加权为None
        self.ensemble_weight = infer_cfg.ensemble_weight
        self.boost_model = infer_cfg.boost_model
        self.pool = infer_cfg.pool  # 进程池
        self.is_model_half = infer_cfg.is_model_half  # 是否采用半精度推理 float16

        self.trans = T.ToTensor()

    # 数据前处理
    def pre_process(self, data):
        # json process
        # json_data = json.loads(data.get_data().decode('utf-8'))
        # img = json_data.get("img")
        # bast64_data = img.encode(encoding='utf-8')
        # print(data.get_data()[-349954:-2] == bast64_data)
        # img = base64.b64decode(data.get_data()[-349954:-2])
        # img = base64.b64decode(bast64_data)
        img = Image.open(BytesIO(bytearray(base64.b64decode(data.get_data()[-349954:-2]))))
        return self.trans(img).to(self.device)
        # img = np.array(img, dtype=np.float32)
        # img = torch.Tensor(img).to(self.device)
        # return img  # 应该是256*256*4

    # 数据后处理
    def post_process(self, data):
        data = data.cpu().data.numpy()
        img_encode = np.array(cv2.imencode('.png', data)[1]).tobytes()
        bast64_data = base64.b64encode(img_encode)
        bast64_str = str(bast64_data, 'utf-8')
        return bast64_str

    def predict(self, data):
        '''
        并行推理
        '''
        # data = data.permute(2, 0, 1)
        # data = torch.unsqueeze(data, dim=0).div(255.0)  # 1*4*256*256
        data = torch.unsqueeze(data, dim=0)  # 1*4*256*256
        data = self.test_transform(data)
        self.data = data.half()
        with torch.no_grad():
            if self.models_num == 1:
                ret = self.models[0](self.data)
                ret = torch.argmax(ret, dim=1)[0] + 1  # 需要1~10
            else:
                # 起多个进程
                out_list = self.pool.map(self._predict, list(range(self.models_num)))
                ret = torch.argmax(sum(out_list), dim=1)[0] + 1  # 需要1~10
        return ret

    def _predict(self, model_num, dest_device='cuda:0'):
        # model = self.models[model_num]
        # model.eval()
        # data = self.data.detach().to(self.device_list[model_num])
        weight = self.ensemble_weight[model_num]
        with torch.no_grad():
            out = torch.nn.functional.softmax(self.models(self.data), dim=1) * weight
        return out

    # 模型预测：默认执行self.model(preprocess_data)，一般不用重写
    # 如需自定义，可覆盖重写

    # 串行推理
    # def predict_serial(self, data):
    #     '''
    #     串行推理
    #     '''
    #     # data.shape应该是256*256*4, np.array
    #
    #     data = data.permute(2, 0, 1)
    #     # data = torch.unsqueeze(data, dim=0)  # 1*4*256*256
    #     data = torch.unsqueeze(data, dim=0) / 255.0  # 1*4*256*256
    #     data = self.test_transform(data)
    #     # data = self.test_transform(data)  # 4*256*256
    #     # data = torch.unsqueeze(data, dim=0)  # 1*4*256*256
    #
    #     out_avg = torch.zeros(size=(len(data), 10, 256, 256)).to('cuda:0')
    #     model_num = 0
    #     with torch.no_grad():
    #         for model in self.models:
    #             temp_out = model(data)
    #             temp_out = torch.nn.functional.softmax(temp_out, dim=1)  # 转成概率
    #             out_avg += self.ensemble_weight[model_num] * temp_out
    #             model_num += 1
    #         ret = torch.argmax(out_avg, dim=1)[0] + 1  # 需要1~10
    #     return ret

    # def parrel_predict(self, data):
    #     data = data.permute(2, 0, 1)
    #     data = torch.unsqueeze(data, dim=0) / 255.0  # 1*4*256*256
    #     data = self.test_transform(data)
    #     thread_list = []
    #     model_num = 0
    #     with torch.no_grad():
    #         for model in self.models:
    #             device = self.device_list[model_num]
    #             weight = self.ensemble_weight[model_num]
    #             temp_data = data.detach().to(device)
    #             pred_thread = PredictThread(data=temp_data,
    #                                         model=model,
    #                                         weight=weight,
    #                                         dest_device='cuda:0')
    #             thread_list.append(pred_thread)
    #             pred_thread.start()
    #             model_num += 1
    #
    #         out_list = []
    #         for thread in thread_list:
    #             thread.join()  # 一定要join，不然主线程比子线程跑的快，会拿不到结果
    #             out_list.append(thread.get_result())
    #         ret = torch.argmax(sum(out_list), dim=1)[0]
    #     return ret
    #
    # def _predict(self, data, model, weight, dest_device='cuda:0'):
    #     '''
    #     单个进程的预测
    #     '''
    #     out = model(data) * weight
    #     return out.to(dest_device)

# class PredictThread(threading.Thread):
#     def __init__(self, data, model, weight, dest_device='cuda:0'):
#         super(PredictThread, self).__init__()
#         self.data = data
#         self.model = model
#         self.weight = weight
#         self.dest_device = dest_device
#
#     def _predict(self):
#         '''
#         单个进程的预测
#         '''
#         out = self.model(self.data) * self.weight
#         return out.to(self.dest_device)
#
#     def run(self):
#         self.result = self._predict()
#
#     def get_result(self):
#         return self.result
