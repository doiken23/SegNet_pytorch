##########################################
##### This is the code for inference #####
##### University of Tokyo Doi Kento  #####
##########################################

# import PyTorch modele
import torch
import torch.nn as nn
from torch.autograd import Variable
import segnet

# import python modele
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from tqdm import tqdm
import argparse
from PIL import Image
import math

# get arguments
def get_arguments():
    parser = argparse.ArgumentParser(description='Pytorch inference tool via SegNet')
    parser.add_argument('trained_model', type=str, help='path of trained parameter')
    parser.add_argument('image', type=str, help='path of image')
    parser.add_argument('outdir', type=str, help='path of out dir')
    parser.add_argument('band_num', type=int, help='band number')
    parser.add_argument('class_num', type=int, help='class number')
    parser.add_argument('window_size', type=int, help='window size')
    parser.add_argument('overlap', type=int, help='over lap size')
    args = parser.parse_args()
    return args

def main():
    # get arguments
    args = get_arguments()

    # load the image
    if args.image[-3:] == 'tif':
        image_array = np.array(Image.open(args.image))
    elif args.image[-3:] == 'npy':
        image_array = np.load(args.image)
    else:
        image_array = np.array(Image.open(args.image))

    c, h, w = image_array.shape

    # compute the patch size
    w_size = args.window_size
    ol = args.overlap
    Y = math.ceil((h - w_size) / (w_size - ol))
    X = math.ceil((w - w_size) / (w_size - ol))
    out_h = w_size + (w_size - ol)* Y
    out_w = w_size + (w_size - ol)* X

    # prepare input and output array
    input_tensor = torch.zeros((args.band_num, out_h, out_w))
    input_tensor[:,0:h ,0:w] = torch.Tensor(image_array[:, 0:h, 0:w].astype(np.float))
    input_tensor = Variable((input_tensor.float()/255.0).cuda(), volatile=True)
    net_output = torch.zeros((args.class_num ,w_size + (w_size - ol)*Y, w_size + (w_size - ol)*X))
    inferenced_array = np.zeros((1, h, w))

    # load the trained network
    net = segnet.SegNet(args.band_num, args.class_num,0)
    weight = torch.load(args.trained_model)
    net.load_state_dict(weight)
    net.eval().cuda()

    # do inference (main loop)
    print('***** start inference *****')
    for i in tqdm(np.arange(Y)):
        for j in np.arange(X):
            y = (w_size-ol) * i
            x = (w_size-ol) * j
            input = input_tensor[:, y: y+w_size, x: x+w_size]
            net_output[:, y: y+w_size, x: x+w_size] = torch.add(net_output[:, y: y+w_size, x: x+w_size], net(input.contiguous().view(1, input_tensor.size()[0], w_size, w_size)).data.squeeze().cpu())

    # make and save inferenced image
    _, inferenced_array = torch.max(net_output[:, 0:h, 0:w], 0)
    inferenced_array = inferenced_array.numpy().astype(np.uint8)
    Image.fromarray(inferenced_array).save(os.path.join(args.outdir, args.image.split('\\')[-1][:-4] + '_inferenced.png'))

# ------------------- #
# main program
if __name__ == '__main__':
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print('elapsed time: {} [sec]'.format(elapsed_time))
