##########################################
##### This is the code for inference #####
##### University of Tokyo Doi Kento  #####
##########################################

# import PyTorch modele
import torch
import torch.nn.functional as F
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
    model_path = args.trained_model
    image_dir = args.image
    out_dir = args.outdir
    band_num = args.band_num
    class_num = args.class_num
    w_size = args.window_size
    ol = args.overlap

    # make iamge path list
    if os.path.isdir(image_dir):
        image_path_list = os.listdir(image_dir)
        image_path_list = [os.path.join(image_dir, image_path) for image_path in image_path_list]
    else:
        image_path_list = [image_dir]

    # prepare the network
    print("loading the network")
    net = segnet.SegNet(band_num, class_num, 0)
    weight = torch.load(model_path)
    net.load_state_dict(weight)
    net.eval().cuda()

    # do inferences
    for image_path in tqdm(image_path_list):
        do_inference(net, image_path, band_num, class_num, w_size, ol, out_dir)


def do_inference(net, image_path, band_num, class_num, w_size, ol, out_dir):
    """
    image_path     :path of input numpy array
    band_num       :band number of input image
    w_size         :window size
    ol             :overlap number
    out_dir        :output directory
    """
    # load the image array
    image_array = np.load(image_path)

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
    output_tensor = torch.zeros((class_num ,w_size + (w_size - ol)*Y, w_size + (w_size - ol)*X)).cuda()
    confidence_map = np.zeros((1, h, w))
    label_map = np.zeros((1, h, w))

    # do inference (main loop)
    print('***** start inference *****')
    for i in tqdm(np.arange(Y+1)):
        for j in np.arange(X+1):
            y = (w_size-ol) * i
            x = (w_size-ol) * j
            input = input_tensor[:, y: y+w_size, x: x+w_size]
            output = net(input.contiguous().view(1, input_tensor.size()[0], w_size, w_size))
            output_tensor[:, y: y+w_size, x: x+w_size] = torch.add(output_tensor[:, y: y+w_size, x: x+w_size], output.data.squeeze())

    # make and save inferenced image
    confidence_tensor = F.softmax(output_tensor).squeeze()
    confidence_map, label_map = torch.max(confidence_tensor[:, 0:h, 0:w], 0)
    confidence_map = confidence_map.data.cpu().numpy()
    label_map = label_map.data.cpu().numpy().astype(np.uint8)

    np.save(os.path.join(out_dir, image_path.split('\\')[-1][:-4] + '_confidence'), confidence_map)
    Image.fromarray(label_map).save(os.path.join(out_dir, image_path.split('\\')[-1][:-4] + '_inferenced_label.png'))

# ------------------- #
# main program
if __name__ == '__main__':
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print('elapsed time: {} [sec]'.format(elapsed_time))
