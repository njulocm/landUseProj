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
from addict import Dict

import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
from tensorrt.tensorrt import Logger
from utils import torch2trt

from model import build_model, EnsembleModel

# from ptflops import get_model_complexity_info
import _thread
import threading
# from multiprocessing import Pool  # 需要针对 IO 密集型任务和 CPU 密集型任务来选择不同的库
from multiprocessing.dummy import Pool  # 需要针对 IO 密集型任务和 CPU 密集型任务来选择不同的库
import time

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # 多卡训练，要设置可见的卡，device设为cuda即可，单卡直接注释掉，device正常设置
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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
    processes_num = test_cfg.setdefault(key='processes_num', default=4)
    pool = Pool(processes=processes_num)  # 进程数可能需要调节
    device = test_cfg.setdefault(key='device', default='cuda:0')
    # device_available = test_cfg.setdefault(key='device_available', default=['cuda:0'])
    is_model_half = test_cfg.setdefault(key='is_model_half', default=False)
    # 获取模型集成的权重
    ensemble_weight = None
    ensemble_weight = test_cfg.setdefault(key='ensemble_weight',
                                          default=[1.0 / len(test_cfg.check_point_file)] * len(
                                              test_cfg.check_point_file))
    if len(ensemble_weight) != len(test_cfg.check_point_file):
        raise Exception('权重个数错误！')

    # TRT相关
    is_trt_infer = test_cfg.setdefault(key='is_trt_infer', default=False)
    FLOAT = test_cfg.setdefault(key='FLOAT', default=32)

    # torch 推理相关参数初始化
    models = []
    device_list = []  # 需要考虑并行的方法
    models_num = 0

    # trt推理相关参数初始化
    context = None
    inputs = None
    outputs = None
    bindings = None
    stream = None

    if is_trt_infer:  # trt推理，目前只支持单模
        # 把ckpt转为trt
        TRT_LOGGER = trt.Logger(min_severity=Logger.ERROR)
        print("正在把多模型转为集成模型...")
        print(test_cfg.check_point_file)
        ensemble_model = EnsembleModel(check_point_file_list=test_cfg.check_point_file, device=device)
        if not os.path.exists('../user_data/checkpoint/ensemble'):
            os.system("mkdir ../user_data/checkpoint/ensemble")
        ensemble_ckpt_file = '../user_data/checkpoint/ensemble/model.pth'
        trt_file = '../user_data/checkpoint/ensemble/model.trt'
        if not os.path.exists(trt_file):
            torch.save(ensemble_model, ensemble_ckpt_file)
            torch2trt(ckpt_path=ensemble_ckpt_file, FLOAT=FLOAT)

        # 加载trt
        asshole = torch.ones(1).cuda()  # 这玩意儿没用，只是用来把cuda初始化一下，不然会报错
        # Build an engine
        print(trt_file)
        with open(trt_file, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        # Create the context for this engine
        context = engine.create_execution_context()
        # Allocate buffers for input and output
        inputs, outputs, bindings, stream = allocate_buffers(engine)  # input, output: host # bindings

    else:  # torch推理
        # 加载多个模型
        for ckpt in test_cfg.check_point_file:
            # device = device_available[models_num % len(device_available)]
            model = torch.load(ckpt, map_location=device)
            print(ckpt)
            # model_cfg = cfg.model_cfg
            # model = build_model(model_cfg).to(device)
            if isinstance(model, torch.nn.DataParallel):  # 如果是双卡模型，需要去除module
                model = model.module
            model.eval()
            models.append(model)
            device_list.append(device)
            models_num += 1

    # 构建model_dict
    infer_cfg = Dict(
        {'models': models,
         'models_num': models_num,
         # 'device_list': device_list,
         'device': device,
         'test_transform': test_transform,
         'ensemble_weight': ensemble_weight,
         'pool': pool,  # 进程池
         'is_model_half': is_model_half,  # 是否采用半精度推理 float16

         # trt相关参数
         'is_trt_infer': is_trt_infer,
         'context': context,
         'inputs': inputs,
         'outputs': outputs,
         'bindings': bindings,
         'stream': stream,
         'FLOAT': FLOAT,
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
        self.ensemble_weight = infer_cfg.ensemble_weight
        self.pool = infer_cfg.pool  # 进程池
        self.is_model_half = infer_cfg.is_model_half  # 是否采用半精度推理 float16

        self.trans = T.ToTensor()

        # trt相关参数
        self.is_trt_infer = infer_cfg.is_trt_infer
        self.context = infer_cfg.context
        self.inputs = infer_cfg.inputs
        self.outputs = infer_cfg.outputs
        self.bindings = infer_cfg.bindings
        self.stream = infer_cfg.stream
        self.FLOAT = infer_cfg.FLOAT

        self.init_tensorrt()

    def init_tensorrt(self):
        for i in range(1000):
            self.inputs[0].host = np.ones((4, 256, 256), dtype=np.float32)
            do_inference(context=self.context, bindings=self.bindings, inputs=self.inputs,
                         outputs=self.outputs, stream=self.stream)[0].reshape((256, 256))

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
        img = np.array(img, dtype=np.float32)
        return img

    # 数据后处理
    def post_process(self, data):
        img_encode = np.array(cv2.imencode('.png', data)[1]).tobytes()
        bast64_data = base64.b64encode(img_encode)
        bast64_str = str(bast64_data, 'utf-8')
        return bast64_str

    def predict(self,  data):
        # data = self.test_transform(data)  # 4*256*256
        # self.inputs[0].host = data.cpu().numpy()
        self.inputs[0].host = data
        ret = do_inference(context=self.context, bindings=self.bindings, inputs=self.inputs,
                                  outputs=self.outputs, stream=self.stream)[0].reshape((256, 256))
        return ret

    # def predict(self, data):
    #     '''
    #     并行推理
    #     '''
    #     # data = data.permute(2, 0, 1)
    #     # data = torch.unsqueeze(data, dim=0).div(255.0)  # 1*4*256*256
    #     data = data.to(self.device)
    #     data = torch.unsqueeze(data, dim=0)  # 1*4*256*256
    #     data = self.test_transform(data)
    #     self.data = data
    #     with torch.no_grad():
    #         if self.models_num == 1:
    #             ret = self.models[0](self.data)
    #             ret = torch.argmax(ret, dim=1)[0] + 1  # 需要1~10
    #         else:
    #             # 起多个进程
    #             out_list = self.pool.map(self._predict, list(range(self.models_num)))
    #             ret = torch.argmax(sum(out_list), dim=1)[0] + 1  # 需要1~10
    #     return ret.cpu().data.numpy()
    #
    # def _predict(self, model_num, dest_device='cuda:0'):
    #     # model = self.models[model_num]
    #     # model.eval()
    #     # data = self.data.detach().to(self.device_list[model_num])
    #     weight = self.ensemble_weight[model_num]
    #     with torch.no_grad():
    #         out = torch.nn.functional.softmax(self.models(self.data), dim=1) * weight
    #     return out


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        """Within this context, host_mom means the cpu memory and device means the GPU memory
        """
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer data from CPU to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]
