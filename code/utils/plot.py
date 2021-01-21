import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors

# 配置colormap
color_dict = {'farmland': 'springgreen',
              'forest': 'forestgreen',
              'grass': 'greenyellow',
              'road': 'gold',
              'urban': 'red',
              'countyside': 'pink',
              'industry': 'darkslategray',
              'construction': 'silver',
              'water': 'aqua',
              'bareland': 'saddlebrown'
              }
cmap = colors.ListedColormap(list(color_dict.values()), 'indexed')


def plot_picture(pred, label, save_path=None, is_show=False):
    # 散点图
    x = np.array([list(range(256))] * 256).flatten().tolist()
    y = np.array([[i] * 256 for i in range(256)]).flatten().tolist()

    # 增加一些值，充满颜色空间
    xx = [256] * 10
    yy = [256] * 10
    cc = list(range(1, 11))

    fig = plt.figure(figsize=(10, 4))
    # 绘制预测图
    plt.subplot(1, 2, 1)
    plt.title('pred')
    ax = plt.gca()
    plt.scatter(x + xx, y + yy, c=pred.flatten().tolist() + cc, cmap=cmap)
    cbar = plt.colorbar()
    cbar.set_ticks(np.linspace(1.45, 9.55, 10))
    cbar.set_ticklabels(list(color_dict.keys()))
    plt.xlim((0, 255))
    plt.ylim((0, 255))
    ax.invert_yaxis()

    # 绘制实际图
    plt.subplot(1, 2, 2)
    plt.title('label')
    ax = plt.gca()
    plt.scatter(x + xx, y + yy, c=label.flatten().tolist() + cc, cmap=cmap)
    cbar = plt.colorbar()
    cbar.set_ticks(np.linspace(1.45, 9.55, 10))
    cbar.set_ticklabels(list(color_dict.keys()))
    plt.xlim((0, 255))
    plt.ylim((0, 255))
    ax.invert_yaxis()

    # 保存图片
    if save_path:
        plt.savefig(save_path)
        print('已保存'+save_path)

    # 显示图片
    if is_show:
        plt.show()


if __name__ == '__main__':
    pred_dir = '/home/cm/landUseProj/prediction_result/Unet_val_out/'
    label_dir = '/home/cm/landUseProj/tcdata/suichang_round1_train_210120/'
    save_dir = '/home/cm/landUseProj/user_data/结果比较/Unet/'

    pic_list = os.listdir(pred_dir)
    pic_list.sort()

    for index in range(len(pic_list)):
        pic_name = pic_list[index]
        pred = cv2.imread(pred_dir + pic_name, cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(label_dir + pic_name, cv2.IMREAD_GRAYSCALE)
        plot_picture(pred, label, save_path=save_dir + pic_name, is_show=False)

    print('end')
