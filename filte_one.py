#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Liang
"""
import numpy as np
import options
import unet
import torch
import visdom
import pickle
import os
from train_unet import fill_image, preprocess
import argparse
from datasets import Sino_Dataset
import matplotlib.pylab as plt
from torch.autograd import Variable

opts = options.parse()

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, dest='model_dict', default='/home/liang/Desktop/output_motions/model/model_dict_0',
                        help='file of model dict')
parser.add_argument('--valid', type=str, dest='valid', default='/home/liang/Desktop/test/0_dataset.pkl',
                        help='file of validiation sinogram')
parser.add_argument('--output-pickle', type=str, dest='output_pickle', default="/home/liang/Desktop/test/motion_image.pkl",
                        help='file of validiation sinogram')
parser.add_argument('--output-sino', type=str, dest='output_sino', default="/home/liang/Desktop/test/motion_image.v",
                        help='file of validiation sinogram')

args = parser.parse_args()

network = unet.Sino_repair_net(opts, [0, 1])
network.network.load_state_dict(torch.load(args.model_dict))
network.network.eval()

file_path = os.path.join(opts.output_path, "file")
if not file_path:
    os.makedirs(file_path)



vis = visdom.Visdom()
window = None
window2 = None
show = True
output = []
scalerunet = 5
output_list = []

batchsize = 1 # can divide 2*815

datafile = '/home/liang/Desktop/motion_pickle/p1emoco_gated_000_000_00.pkl'
data = Sino_Dataset(datafile, 100000, testing=True, input_depth=opts.input_channel_number, output_depth=opts.output_channel_number, is_test_in_train=False)
train_loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True,
                                           drop_last=True)
for i, (good, bad, valid) in enumerate(train_loader):

    print('_slice_ {}'.format(str(i)))

    def thread(good, bad, valid):
        target_img, corrupt_img = good.cuda(), bad.cuda()  # 1*64*520

        scaler, corrupt_img, target_img = preprocess(corrupt_img, target_img)
        corrupt_img, target_img = corrupt_img.type(torch.float32), target_img.type(torch.torch.float32)
        out, loss = network.test(corrupt_img, target_img, valid)  # 1*1*64*520

        out = out.cpu().detach().numpy() *scaler[0][1].item() # .reshape((batchsize, -1, 64, 520)) # 1*64*520
        print(out.shape)

        return out, out.astype('float32')

    out, output_save = thread(good, bad, valid)

    # if show:
    #     step = 4
    #     final = out
    #     different = abs(final-good.cpu().numpy())
    #     images = np.stack(
    #         [(good[0][:1] / (good[0][:1].max() + 1e-8) * 255)[:, ::step, ::step],
    #          (final[0][:1] / (final[0][:1].max() + 1e-8) * 255)[:, ::step, ::step],
    #          (out[0][:1] / (out[0][:1].max() + 1e-8) * 255)[:, ::step, ::step],
    #          (bad[0][:1] / (bad[0][:1].max() + 1e-8) * 255)[:, ::step, ::step],
    #          (different[0][:1] / (different[0][:1].max() + 1e-8) * 255)[:, ::step, ::step]])  # 1*64*520
    #     # images = np.stack([good[0][:1] / good[0][:1].max() * 255, final[0][:1] / final[0][:1].max() * 255,
    #     #                    out[0][:1] / out[0][:1].max() * 255, bad[0][:1] / bad[0][:1].max() * 255])  # 1*64*520
    #     if not window2:
    #         window2 = vis.images(images, padding=5, nrow=1, opts=dict(title='results: Good, Final, Output, Bad'))
    #     else:
    #         vis.images(images, padding=5, win=window2, nrow=1, opts=dict(title='results: Good, Final, Output, Bad'))

    for j in range(batchsize):
        output_list.append(output_save[j])

    # for j in range(batchsize):
    #     output_list.append(1)


output = np.stack(output_list)
data = output.reshape(-1, 220, 220)
data = data[1:, :, :]
# save the file in case
with open(args.output_pickle, 'wb') as f:
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

print(data.shape)
data = data.flatten()
data.tofile(args.output_sino)

