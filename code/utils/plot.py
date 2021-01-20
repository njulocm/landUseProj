import os
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import gridspec


def plot_picture(pred, label):
    # 第一种画法
    plt.subplot(1, 2, 1)
    plt.imshow(pred, cmap='Paired')
    plt.title('pred')
    plt.subplot(1, 2, 2)
    plt.imshow(label, cmap='Paired')
    plt.title('label')
    plt.legend()
    plt.show()

    # 第二种画法
    # cMap = ListedColormap(['white', 'green', 'blue', 'red', 'white', 'green', 'blue', 'red', 'white', 'green'])
    # # gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.2])
    # # ax1 = gs[0]
    # # ax2 = gs[1]
    #
    # fig, (ax1, ax2) = plt.subplots(1, 2, sharex='all', figsize=(30, 15))
    # # ig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(45, 15))
    # heatmap1 = ax1.pcolor(pred, cmap=cMap)
    # heatmap2 = ax2.pcolor(label, cmap=cMap)
    # cbar = plt.colorbar(heatmap1)
    # # cbar.ax.get_yaxis().set_ticks([])
    # # cbar.ax.set_yticklabels(['0', '1', '2', '>3'])
    # # for j, lab in enumerate(['$0$', '$1$', '$2$', '$>3$']):
    # #     cbar.ax.text(.5, (2 * j + 1) / 8.0, lab, ha='center', va='center')
    # # cbar.ax.get_yaxis().labelpad = 15
    #
    # plt.show()


if __name__ == '__main__':
    pred_dir = '/home/cm/landUseProj/prediction_result/Unet_val_out/'
    label_dir = '/home/cm/landUseProj/tcdata/suichang_round1_train_210120/'

    pic_list = os.listdir(pred_dir)
    pic_list.sort()

    index = 150
    pic_name = pic_list[index]
    pred = cv2.imread(pred_dir + pic_name, cv2.IMREAD_GRAYSCALE)
    label = cv2.imread(label_dir + pic_name, cv2.IMREAD_GRAYSCALE)
    plot_picture(pred, label)

    print('end')
