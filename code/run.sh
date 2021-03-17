#!/bin/bash
nvidia-smi
python main.py ./config/smp_unetpp_config.py test_online
