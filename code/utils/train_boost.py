import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import xgboost
import joblib


def train_adaBoost_main(feature, label, ckpt_file):
    model = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),
        n_estimators=100, learning_rate=1, random_state=6666)
    model.fit(feature, label)
    # out = model.predict(feature)
    print('模型得分：{}'.format(model.score(feature, label)))
    # 保存模型
    joblib.dump(model, ckpt_file)


def train_XGboost_main(feature, label, ckpt_file):
    params = {
        'booster': 'gbtree',
        'objective': 'multi:softmax',  # 多分类问题
        'num_class': 10,  # 类别数，与multi softmax并用
        'eta': 0.07,  # 如同学习率
        # 'gamma': 0.1,  # 用于控制是否后剪枝的参数，越大越保守，一般0.1 0.2的样子
        'max_depth': 12,  # 构建树的深度，越大越容易过拟合
        'subsample': 0.7,  # 随机采样训练样本
        'colsample_bytree': 1,  # 这个参数默认为1，是每个叶子里面h的和至少是多少
        # 'min_child_weight': 1,  # 对于正负样本不均衡时的0-1分类而言，假设h在0.01附近，min_child_weight为1
        # 意味着叶子节点中最少需要包含100个样本。这个参数非常影响结果，
        # 控制叶子节点中二阶导的和的最小值，该参数值越小，越容易过拟合
        "scale_pos_weight": 10,  # 大于0的取值可以处理类别不平衡的情况。帮助模型更快收敛
        'silent': 0,  # 设置成1 则没有运行信息输入，最好是设置成0
        'seed': 6666,
        # 'nthread': 7,  # CPU线程数，不设置则为最大
        'eval_metric': 'merror'
    }
    # feature = np.argmax(feature.reshape((feature.shape[0],14,10)),axis=2)

    val_num = 950000 # 验证集数目
    data_train = feature[:-val_num, :]
    label_train = label[:-val_num]
    data_val = feature[-val_num:, :]
    label_val = label[-val_num:]
    dtrain = xgboost.DMatrix(data=data_train, label=label_train)
    dval = xgboost.DMatrix(data=data_val, label=label_val)
    xgb_model = xgboost.train(params, dtrain=dtrain, num_boost_round=1000, evals=[(dtrain, 'train'), (dval, 'val')],
                              early_stopping_rounds=100,
                              verbose_eval=True)
    xgb_model.save_model(ckpt_file)

    # tar = xgb.Booster(model_file='xgb.model')


def compute_weight_accuracy(feature, label):
    feature = feature.reshape((-1, 14, 10))
    feature = feature.transpose(0, 2, 1)
    weight = np.array([0.2 / 5] * 5 + [0.2 / 5] * 5 + [0.6 / 4] * 4)
    pred = np.matmul(feature, weight)
    pred = pred.argmax(axis=1)
    acc = (pred==label).sum()/len(pred)
    print(acc)


if __name__ == "__main__":
    feature = np.load('/home/cm/landUseProj/user_data/feature_sample200.npy')
    land_class = np.load('/home/cm/landUseProj/user_data/land_class_sample200.npy')

    compute_weight_accuracy(feature, land_class)

    ckpt_file = '/home/cm/landUseProj/code/checkpoint/adaBoost/xgBoost_b6_b7_other_sample200_iter1000.pkl'
    # train_adaBoost_main(feature=feature, label=land_class, ckpt_file=ckpt_file)

    # model = xgboost.Booster(model_file=ckpt_file)
    # out = model.predict(xgboost.DMatrix(data=feature))
    train_XGboost_main(feature=feature, label=land_class, ckpt_file=ckpt_file)
