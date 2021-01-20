import numpy as np
from opts import TrainOptions

opt = TrainOptions().parse()
dataset = np.load(opt.dataset)['data'][:, :, 0]
pre_len = 12 if "8" in dataset else 9
predict_length = int(opt.hdwps.split(',')[-2])
print(dataset.shape)

def ave_last_12(dataset, time_step):
    train_len = dataset[-24 * 12 * pre_len - 2 * time_step - 24 * 12 * 7: - 24 * 12 * 7 ,:]
    test_y = dataset[-24 * 12 * pre_len - time_step: , :]
    
    print(train_len.shape, test_y.shape)

    MSE_LIST = []
    RMSE_LIST =[]
    MAE_LIST = []
    outlist = []

    for i in range(len(test_y)):
        output = np.mean(train_len[i:i+time_step,:],axis = 0)
        output = np.expand_dims(output, axis = 0)
        outlist.append(output)

    for i in range(len(test_y)):
        out = outlist[i:i+time_step]
        label = test_y[i:i+time_step]
        MSE = np.mean(np.power((out - label), 2))
        RMSE = np.sqrt(np.mean(np.power((out - label), 2)))
        MAE = np.mean(np.abs(out - label))

        MSE_LIST.append(MSE)
        RMSE_LIST.append(RMSE)
        MAE_LIST.append(MAE)

    MSE = np.mean(MSE_LIST)
    RMSE = np.mean(RMSE_LIST)
    MAE = np.mean(MAE_LIST)

    print("FINAL MSE: {:.2f} , RMSE: {:.2f} , MAE: {:.2f}".format(MSE,RMSE,MAE))

if __name__ == "__main__":
    ave_last_12(dataset,predict_length)