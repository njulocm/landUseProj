#!/bin/bash
#time1=$(date +%s -d '1990-01-01 01:01:01');
#python main.py ./config/online_config.py train
#time2=$(date +%s -d '1990-01-01 01:01:01');
#python main.py ./config/online_swa_config.py train
#time3=$(date +%s -d '1990-01-01 01:01:01');
#python main.py ./config/online_config.py test_online
#time4=$(date +%s -d '1990-01-01 01:01:01');
#time_train=$(($time2-$time1));
#time_trainswa=$(($time3-$time2));
#time_test=$(($time4-$time3));
#echo $time_train;
#echo $time_trainswa;
#echo $time_test;

python main.py ./config/smp_unetpp_config.py test_online

#python main.py ./config/smp_unetpp_parallel_config.py test_online
