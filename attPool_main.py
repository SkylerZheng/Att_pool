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

#
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# define attention module
class AttModule(nn.Module):
    def __init__(self):
        super(AttModule, self).__init__()
        self.fc0 = nn.Linear(6, 1)
    def forward(self, x):
        a = x.permute(0, 2, 3, 1).contiguous().view(-1, x.size()[1])
        b = F.relu(self.fc0(a)).squeeze(1)
        c = b.view(x.size()[0], x.size()[2], x.size()[3]).unsqueeze(1).repeat(1, x.size()[1], 1, 1)
        d = torch.mul(x, c)
        return d

# wrap the attention module as a function for easy call
# def AttPool(input_):
#     AttPool = AttModule()(input_)
#     return AttPool.cuda()

AttPool = AttModule()
AttPool.cuda()
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.pool1 = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # self.fc0 = nn.Linear(6, 1)
        # self.fc01 = nn.Linear(16, 1)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        # x = AttModule(x)
        x = AttPool(x)
        y = self.pool(x)
        # # baseline
        # y = self.pool(F.relu(x))
        y = self.conv2(y)
        # y = AttModule(y)
        z = self.pool(F.relu(y))
        # # baseline
        # z = self.pool(F.relu(y))
        out = z.view(-1, 16 * 5 * 5)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.pool1 = nn.AvgPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc0 = nn.Linear(6, 1)
#         self.fc01 = nn.Linear(16, 1)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         a = x.permute(0, 2, 3, 1).contiguous().view(-1, x.size()[1])
#         b = F.relu(self.fc0(a)).squeeze(1)
#         c = b.view(x.size()[0], x.size()[2], x.size()[3]).unsqueeze(1).repeat(1, x.size()[1], 1, 1)
#         d = torch.mul(x, c)
#         y = self.pool(F.relu(d))
#         # # baseline
#         # y = self.pool(F.relu(x))
#         y = self.conv2(y)
#         a1 = y.permute(0, 2, 3, 1).contiguous().view(-1, y.size()[1])
#         b1 = F.relu(self.fc01(a1)).squeeze(1)
#         c1 = b1.view(y.size()[0], y.size()[2], y.size()[3]).unsqueeze(1).repeat(1, y.size()[1], 1, 1)
#         d1 = torch.mul(y, c1)
#         z = self.pool(F.relu(d1))
#         # # baseline
#         # z = self.pool(F.relu(y))
#         z = z.view(-1, 16 * 5 * 5)
#         z = F.relu(self.fc1(z))
#         z = F.relu(self.fc2(z))
#         z = self.fc3(z)
#         return z

net = Net()
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

for epoch in range(5):  # loop over the dataset multiple times

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

# ########################################################################
# # That looks waaay better than chance, which is 10% accuracy (randomly picking
# # a class out of 10 classes).
# # Seems like the network learnt something.
# #
# # Hmmm, what are the classes that performed well, and the classes that did
# # not perform well:
#
# class_correct = list(0. for i in range(10))
# class_total = list(0. for i in range(10))
# for data in testloader:
#     images, labels = data
#     images, labels = images.cuda(), labels.cuda()
#     outputs = net(Variable(images))
#     _, predicted = torch.max(outputs.data, 1)
#     c = (predicted == labels).squeeze()
#     for i in range(4):
#         label = labels[i]
#         class_correct[label] += c[i]
#         class_total[label] += 1
#
#
# for i in range(10):
#     print('Accuracy of %5s : %2d %%' % (
#         classes[i], 100 * class_correct[i] / class_total[i]))
