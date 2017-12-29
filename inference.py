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
    model_path = args.trainede_model
    image_path = args.image
    out_dir = args.outdir
    band_num = args.band_num
    class_num = args.class_num
    w_size = args.window_size
    ol = args.overlap

    # make iamge path list
    if os.path.isdir(image_path):
        image_path_list = os.listdir(image_path)
        image_path_list.sort
    else:
        image_path_list = [image_path]

    # prepare the network
    print("loading the network")
    net = segnet.SegNet(band_num, class_num, 0)
    weight = torch.load(model_path)
    net.load_state_dict(weight)
    net.eval().cuda()

    # do inferences
    print("start inference")
    for image_path in tqdm(image_path_list):
        image_array = np.load(image_path)
        do_inference(net, image_array, band_num, w_sizes, ol, out_dir)


def do_inference(net, image_array, bund_num,  w_size, ol, out_dir):
    """
    image_array    :input numpy array
    band_num       :band number of input image
    w_size         :window size
    ol             :overlap number
    out_dir        :output directory
    """
    # compute the patch size
    c, h, w = image_array.shape
    Y = math.ceil((h - w_size) / (w_size - ol))
    X = math.ceil((w - w_size) / (w_size - ol))
    out_h = w_size + (w_size - ol)* Y
    out_w = w_size + (w_size - ol)* X

    # prepare input and output array
    input_tensor = torch.zeros((band_num, out_h, out_w))
    input_tensor[:,0:h ,0:w] = torch.Tensor(image_array[:, 0:h, 0:w].astype(np.float))
    input_tensor = Variable((input_tensor.float() / 255.0).cuda(), volatile=True)
    output_tensor = torch.zeros((1, args.class_num ,w_size + (w_size - ol)*Y, w_size + (w_size - ol)*X)).cuda()
    confidence_map = np.zeros((1, h, w))
    label_map = np.zeros((1, h, w))

    # do inference (main loop)
    print('***** start inference *****')
    for i in tqdm(np.arange(Y)):
        for j in np.arange(X):
            y = (w_size-ol) * i
            x = (w_size-ol) * j
            input = input_tensor[:, y: y+w_size, x: x+w_size]
            output = net(input.contiguous().view(1, input_tensor.size()[0], w_size, w_size))
            output_tensor[:, y: y+w_size, x: x+w_size] = torch.add(net_output[:, y: y+w_size, x: x+w_size], output)

    # make and save inferenced image
    confidence_tensor = nn.Softmax2d(output_tensor).squeeze()
    confidence_map = confidence_tensor[:, 0:h, 0:w].cpu().numpy()
    _, label_map = torch.max(output_tensor.squeeze()[:, 0:h, 0:w], 0)
    label_map = label_map.cpu().numpy().astype(np.uint8)

    Image.fromarray(confidence_map).save(os.path.join(args.outdir, args.image.split('\\')[-1][:-4] + '_confidence.png'))
    Image.fromarray(label_map).save(os.path.join(args.outdir, args.image.split('\\')[-1][:-4] + '_inferenced_label.png'))

# ------------------- #
# main program
if __name__ == '__main__':
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print('elapsed time: {} [sec]'.format(elapsed_time))
