########################################
###### This script is made by Doi Kento.
###### University of Tokyo
########################################

# import torch module
import torch
import torch.utils.data as data_utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import models, datasets
import segnet

# import python module
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import argparse


def get_argument():
    # get the argment
    parser = argparse.ArgumentParser(description='Pytorch SegNet')
    parser.add_argument('--batch-size', type=int, default=10, help='input batch size for training (default:10)')
    parser.add_argument('--epochs', type=int, default=300, help='number of the epoch to train (default:300)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate for training (default:0.01)')
    parser.add_argument('--momentum', type=int, default=10, help='SGD momentum (default:10)')
    parser.add_argument('image_dir_path', type=str, help='the path of image directory (npy)')
    parser.add_argument('GT_dir_path', type=str, help='the path of GT directory (npy)')
    parser.add_argument('pretrained_model_path', type=str, default=None, help='the path of pretrained model')
    args = parser.parse_args()
    return args

class Numpy_Dataset(data_utils.Dataset):
    def __init__(self, img_dir, GT_dir):
        self.img_dir = img_dir
        self.GT_dir = GT_dir
        self.img_list = [os.path.join(img_dir, img_path) for img_path in os.listdir(img_dir)]
        self.GT_list = [os.path.join(GT_dir, GT_path) for GT_path in os.listdir(GT_dir)]

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        GT_path = self.GT_list[idx]
        image_arr = np.load(img_path)
        GT_arr = np.load(GT_path)

        return image_arr, GT_arr

    def __len__(self):
        return len(self.img_list)


def main(args):
    # Loading the dataset
    train_dataset = Numpy_Dataset(os.path.join(args.image_dir_path, 'train'), os.path.join(args.GT_dir_path, 'train'))
    val_dataset = Numpy_Dataset(os.path.join(args.image_dir_path, 'val'), os.path.join(args.GT_dir_path, 'val'))

    train_loader = data_utils.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2)
    val_loader = data_utils.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=1)

    print("Complete the preparing dataset")

    # Define a Loss function and optimizer
    net = segnet.SegNet(3, 5)
    if not args.pretrained_model_path:
        th = torch.load(args.pretrained_model_path).state_dict()
        net.load_state_dict(th)
    net.cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Train the network
    for epoch in range(30):
        net.train()

        running_loss = 0.0
        for i, data in enumerate(train_loader):
            # get the input
            inputs, labels = data
            inputs = inputs.float()
            inputs = inputs / 255.0
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            outputs = outputs.permute(0,2,3,1).contiguous().view(-1, 5).squeeze()

            labels = labels.view(-1).squeeze().long()

            loss = criterion(outputs, labels.view(-1).squeeze())
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if i % 2000 == 1999:
                print ('[%d, %5d] loss: %.df' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

if __name__ == '__main__':
    start = time.time()
    args = get_argument()
    main(args)
    elapsed_time = time.time() - start
    print('elapsed time:{} [sec]'.format(elapsed_time))
