import base64
import requests
import tqdm
import json
import time
import os
import numpy as np
import cv2
from utils.metric import fast_hist,compute_miou
from io import BytesIO
from PIL import Image

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # 多卡训练，要设置可见的卡，device设为cuda即可，单卡直接注释掉，device正常设置
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def loader2json(data):
    base_json = '{"a":1,"b":2,"c":3,"d":4,"e":5}'
    send_json = json.loads(base_json, encoding='utf-8')
    bast64_data = base64.b64encode(data)
    bast64_str = str(bast64_data, 'utf-8')
    send_json['img'] = bast64_str
    send_json = json.dumps(send_json)
    return send_json


def send_eval(data):
    url = "http://127.0.0.1:8080/tccapi"
    start = time.time()
    res = requests.post(url, data, timeout=1000)
    cost_time = time.time() - start
    # res = analysis_res(res)
    return res, cost_time


DIR = '/tcdata/last1000/'
all_cost_time = 0
hist_sum = np.zeros((10, 10))
for i in range(6):
    for filename in tqdm.tqdm(os.listdir(DIR)):
        if filename.split('.')[-1] == 'png':
            continue
        else:
            img_path = os.path.join(DIR, filename)
            fin = open(img_path, 'rb')
            img = fin.read()
            data_json = loader2json(img)
            ret, cost_time = send_eval(data_json)

            ret = np.array(Image.open(BytesIO(bytearray(base64.b64decode(ret.content)))))-1
            # bast64_data = ret.content.encode(encoding='utf-8')
            # ret = base64.b64decode(bast64_data)
            # bytesIO = BytesIO()
            # ret = Image.open(BytesIO(bytearray(ret)))
            # ret = np.array(ret)

            png_path = img_path.split('.tif')[0]+'.png'
            mask = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)-1
            hist = fast_hist(mask, ret, 10)
            hist_sum += hist
            all_cost_time += cost_time

miou = compute_miou(hist_sum)
Fe = 0.4 * (1 - (min(max(all_cost_time, 40.0), 800.0) - 40.0) / (800.0 - 40.0))
print(f'Fe={Fe}')
print(f'miou={miou}')