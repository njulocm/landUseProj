import base64
import requests
import tqdm
import json
import time
import os

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


DIR = '../../tcdata/suichang_round1_test_partA_210120/'
all_cost_time = 0
for i in range(2):
    for filename in tqdm.tqdm(os.listdir(DIR)):
        img_path = os.path.join(DIR, filename)
        fin = open(img_path, 'rb')
        img = fin.read()
        data_json = loader2json(img)
        ret, cost_time = send_eval(data_json)
        all_cost_time += cost_time
Fe = 0.4 * (1 - (min(max(all_cost_time, 40.0), 800.0) - 40.0) / (800.0 - 40.0))
print(Fe)