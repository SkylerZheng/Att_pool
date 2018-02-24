# -*- coding: utf-8 -*-
"""
Training an attentional pooling based CNN for image classification with CIFAR10
"""
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1]

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

########################################################################
# 2. Define a Convolution Neural Network
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Copy the neural network from the Neural Networks section before and modify it to
# take 3-channel images (instead of 1-channel images as it was defined).

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


# define attention module
class AttModule(nn.Module):
    def __init__(self, channels):
        super(AttModule, self).__init__()
        self.fc0 = nn.Linear(channels, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        a = x.permute(0, 2, 3, 1).contiguous().view(-1, x.size()[1])
        # softmax based attpool
        b = F.softmax(self.fc0(a).squeeze(1).view(a.size()[0], -1), dim=1)
        # relu based attpool
        # b = self.relu(self.fc0(a)).squeeze(1)
        c = b.view(x.size()[0], x.size()[2], x.size()[3]).unsqueeze(1).repeat(1, x.size()[1], 1, 1)
        d = torch.mul(x, c)
        e = self.pool(d)
        return e

# define basic blocks

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, stride=stride, bias=False)
class SimpleBlock(nn.Module):
    def __init__(self, inplanes, planes1, planes2, block, stride=1):
        super(SimpleBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes1, stride)
        self.bn1 = nn.BatchNorm2d(planes1)
        self.relu = nn.ReLU(inplace=True)
        # self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(0.5)
        self.conv2 = conv3x3(planes1, planes2)
        self.bn2 = nn.BatchNorm2d(planes2)
        self.layer1 = self._make_layer(block, planes1)
        self.layer2 = self._make_layer(block, planes2)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def _make_layer(self, block, channels):
        layers = []
        layers.append(block(channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.drop(out)
        out = self.layer1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.drop(out)
        out = self.layer2(out)

        out = out.view(-1, out.size()[1] * out.size()[2] * out.size()[3])
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
net = SimpleBlock(3, 64, 128, AttModule)
net.cuda()

########################################################################
# 3. Define a Loss function and optimizer
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Let's use a Classification Cross-Entropy loss and SGD with momentum

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
# optimizer = optim.Adam(net.parameters(), lr=0.001, momentum=0.9)

################################################### #####################
# 4. Train the network
# ^^^^^^^^^^^^^^^^^^^^
#
# This is when things start to get interesting.
# We simply have to loop over our data iterator, and feed the inputs to the
# network and optimize

for epoch in range(20):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if (i+1) % 200 == 0:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0

print('Finished Training')

########################################################################
# 5. Test the network on the test data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
########################################################################

correct = 0
total = 0
for data in testloader:
    images, labels = data
    images, labels = images.cuda(), labels.cuda()
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
