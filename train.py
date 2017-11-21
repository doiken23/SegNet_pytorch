########################################
###### This script is made by Doi Kento.
###### University of Tokyo
########################################

# import torch module
import torch
import torch.utils.data as data_utils
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import models
import segnet

# import python module
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import argparse

# get the argment
parser = argparse.ArgumentParser(description='Pytorch SegNet')
parser.add_argument('--batch-size', type=int, default=10, help='input batch size for training (default:10)')
parser.add_argument('--epochs', type=int, default=300, help='number of the epoch to train (default:300)')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate for training (default:0.01)')
parser.add_argument('--momentum', type=int, default=10, help='SGD momentum (default:10)')
parser.add_argument('train_data_path', type=str, help='the path of training data directory (npy)')
parser.add_argument('validation_data_path', type=str, help='the path of validation data directory (npy)')
args = parser.parse_args()


# Loading the dataset
for i, npy in enumerate(train_data_list):
    data = np.load(npy)
    train_feature_tensor[i,:,:,:] = torch.from_numpy(data)

for i, npy in enumerate(validation_data_list):
    data = np.load(npy)
    train_GT_tensor[i,:,:,:] = torch.from_numpy(data)

# convert the dataset to DataLoader
train = data_utils.TensorDataset(train_feature_tensor, train_GT_tensor)
train_loader = data_utils.DataLoader(train, barch_size=10, shuffle=True)


# Define a Loss function and optimizer
net = segnet.SegNet()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Train the network
for epoch in range(30):

    running_loss = 0.0
    for i, data in enumerate():
        # get the input
        inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss. backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 2000 == 1999:
            print ('[%d, %5d] loss: %.df' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

