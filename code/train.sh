#!/bin/bash

# 训练
python main.py ./config/online_config.py train
python main.py ./config/online_swa_config.py train