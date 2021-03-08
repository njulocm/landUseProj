import numpy as np
from sklearn.ensemble import AdaBoostClassifier
import joblib


def train_adaBoost_main(feature, label, ckpt_file):
    model = AdaBoostClassifier(n_estimators=100, learning_rate=1, random_state=6666)
    model.fit(feature, label)
    # out = model.predict(feature)
    print('模型得分：{}'.format(model.score(feature, label)))
    # 保存模型
    joblib.dump(model, ckpt_file)



if __name__ == "__main__":
    feature = np.load('/home/cm/landUseProj/user_data/feature.npy')
    land_class = np.load('/home/cm/landUseProj/user_data/land_class.npy')
    ckpt_file = '/home/cm/landUseProj/code/checkpoint/adaBoost/adaBoost_b6_b7_others.pkl'
    train_adaBoost_main(feature=feature, label=land_class, ckpt_file=ckpt_file)
