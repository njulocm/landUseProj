#!/bin/bash

# 训练
#python main.py ./config/online_config.py train
#python main.py ./config/online_swa_config.py train

# 预测
#python main.py ./config/online_config.py test_online

python main.py ./config/smp_unetpp_config.py test_online