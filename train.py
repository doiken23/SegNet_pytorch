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
from tqdm import tqdm

def get_argument():
    # get the argment
    parser = argparse.ArgumentParser(description='Pytorch SegNet')
    parser.add_argument('--batch-size', type=int, default=10, help='input batch size for training (default:10)')
    parser.add_argument('--epochs', type=int, default=60, help='number of the epoch to train (default:300)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate for training (default:0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default:0.9)')
    parser.add_argument('image_dir_path', type=str, help='the path of image directory (npy)')
    parser.add_argument('GT_dir_path', type=str, help='the path of GT directory (npy)')
    parser.add_argument('pretrained_model_path', type=str, default=None, help='the path of pretrained model')
    parser.add_argument('--out_path', type=str, default='./weight.pth', help='output weight path')
    args = parser.parse_args()
    return args

class Numpy_Dataset(data_utils.Dataset):
    def __init__(self, img_dir, GT_dir):
        self.img_dir = img_dir
        self.GT_dir = GT_dir
        self.img_list = [os.path.join(img_dir, img_path) for img_path in os.listdir(img_dir)]
        self.GT_list = [os.path.join(GT_dir, GT_path) for GT_path in os.listdir(GT_dir)]
        self.img_list.sort()
        self.GT_list.sort()

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
    loaders = {'train': train_loader, 'val': val_loader}
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
    
    print("Complete the preparing dataset")

    # Define a Loss function and optimizer
    net = segnet.SegNet(3, 5)
    if not args.pretrained_model_path:
        print('load the pretraind mpodel.')
        th = torch.load(args.pretrained_model_path).state_dict()
        net.load_state_dict(th)
    net.cuda()

    criterion = nn.CrossEntropyLoss(size_average=True).cuda()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)

    # initialize the best accuracy and best model weights
    best_model_wts = net.state_dict()
    best_acc = 0.0

    # Train the network
    start_time = time.time()
    for epoch in range(args.epochs):
        print('* ' * 20)
        print('Epoch {}/{}'.format(epoch+1, args.epochs))
        print('* ' * 20)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train(True)
            else:
                net.train(False)

            # initialize the runnig loss and corrects
            running_loss = 0.0
            running_corrects = 0

            for i, data in enumerate(loaders[phase]):
                # get the input
                inputs, labels = data
                inputs = inputs.float()
                inputs = inputs / 255.0

                # wrap the in valiables
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = net(inputs)
                outputs = outputs.permute(0,2,3,1).contiguous().view(-1, 5).squeeze()
                _, preds = torch.max(outputs.data, 1)
                labels = labels.view(-1).squeeze().long()
                loss = criterion(outputs, labels.view(-1).squeeze())

                # backward + optimize if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statuctics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase] 
            epoch_acc = running_corrects / dataset_sizes[phase] / (640*640)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # copy the best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = net.state_dict()

    elapsed_time = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(elapsed_time // 60, elapsed_time % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    net.load_state_dict(best_model_wts)
    return(net)

if __name__ == '__main__':
    args = get_argument()
    model_weights = main(args)
    torch.save(model_weights.state_dict(), args.out_path)
