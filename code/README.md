# 解决方案
本项目使用Unet++模型作为遥感影像分割的baseline，综合使用了swa、crf、集成学习等技术来提升模型的表现。初赛集成了5个模型，在B榜的mIoU为0.4091，具体模型情况如下：
* 采用efficient-b7作为backbone，全数据训练100轮；
* 采用efficient-b7作为backbone，全数据训练100轮，再用crf做后处理；
* 采用efficient-b7作为backbone，全数据训练100轮，再用swa进行精细调参；
* 采用efficient-b7作为backbone，加入attention机制，全数据训练100轮；  
* 采用efficient-b7作为backbone，加入attention机制，用5折交叉验证来训练；
* 对以上5个模型的结果进行加权平均，权重依次为0.15,0.15,0.15,0.15和0.4


# 依赖
本项目使用的是Ubuntu系统，python版本为3.7.9，在训练和测试过程中所用到的python依赖包见requirement.txt文件。

# 训练
依次训练解决方案中提到的5个模型，下面以训练第一个模型为例：
模型的配置文件为code/config/smp_unetpp_config.py，训练时运行如下命令
```
cd landUseProj/code
./train.sh
```

# 测试



