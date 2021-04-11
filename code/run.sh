#!/bin/bash

# -----------------1.划分训练集-----------------
python main.py ./config/online_config.py split_data

# ---------------2.并行训练多模型----------------
# 2.1 有swa的训练
#python main.py ./config/online_config.py train && python main.py ./config/online_swa_config.py train &\
#python main.py ./config/smp_unetpp_color_config.py train && python main.py ./config/smp_unetpp_color_swa_config.py train &
# 2.2 没有swa的训练
python main.py ./config/online_config.py train  &\
python main.py ./config/smp_unetpp_color_config.py train

wait # 等待训练结束

# ----------3.在推理之前打印我们需要的信息----------
nvidia-smi
echo -----------0.40-log-----------
cat ../user_data/log/online.log
echo -----------color-log-----------
cat ../user_data/log/round2_b0_SmpUnetpp_color_alldata-0406.log

#ls ../user_data/round2_val/suichang_round1_train_210120/ | head -n 40

# --------------------4.推理--------------------
python main.py ./config/online_config.py test_online

#python main.py ./config/smp_unetpp_config.py test_online

