import torch
import os
import numpy as np
import re
import pickle
def process_data(file):

    data = np.fromfile(file, dtype='float32')
    data = data.reshape((-1, 220, 220))  # Reshape to the file layout

    return data

path = '/home/liang/Desktop/MoCo_data/'
scans = os.listdir(path)

for folder in scans:


    for file in os.listdir(path+folder+'/'):
        # print(file)
        if re.match('^emoco_gated_000_000_0[0-9]\.v$', file):
            good = path+folder+'/' + file
            corrupted = good.replace('gated','static')
            data_corrupted = process_data(corrupted)
            data_good = process_data(good)
            # data_good = np.expand_dims(data_good, axis=0)
            # data_corrupted = np.expand_dims(data_corrupted, axis=0)

            output_sino_xy = np.stack((data_good, data_corrupted))
            with open("/home/liang/Desktop/motion_pickle/{}.pkl".format(folder+file[:-2]), 'wb') as f:
                pickle.dump(output_sino_xy, f, pickle.HIGHEST_PROTOCOL)
            print(output_sino_xy.shape)
            # print(data_corrupted.shape)
            # print(data_good.shape)
print('finish')
# print(scans)