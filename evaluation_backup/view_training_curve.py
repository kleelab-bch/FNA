import matplotlib
matplotlib.use('agg')
import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    folder_path = 'average_hist/history'
    # for history_file in os.listdir(folder_path):
    history_file = 'all-patients'
    print(folder_path + '/' + history_file)
    data = np.load(folder_path + '/' + history_file, allow_pickle=True, encoding="bytes")
    data_dict = data.ravel()[0]
    if history_file=='history_all-patients.npy':
        print('dice_coef', data_dict['dice_coef'])
        print('val_dice_coef', data_dict['val_dice_coef'])
    for key in data_dict:
        x_coord = list(range(1,len(data_dict[key])+1))
        plt.plot(x_coord, data_dict[key], label=key)
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.title('Training vU-Net')
    plt.savefig(history_file + '_training_graph.png')
    plt.clf()