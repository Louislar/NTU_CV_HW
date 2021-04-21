import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Training: 10 classes, total 40000 images, 4000 in each class 
Validation: 10 classes, total 10000 images, 1000 in each class 

Input: 28*28*1 image
'''

class ConvNet(nn.Module):
    '''
    LeNet-5

    Where's the batch size? --> at data.py and batch size is 32 

    Base model is tooooo good --> acc: 0.996
    How to improve this? 
    '''
    def __init__(self):
        super(ConvNet, self).__init__()
        # TODO
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # TODO
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def name(self):
        return "ConvNet"

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        # TODO

    def forward(self, x):
        # TODO
        return out

    def name(self):
        return "MyNet"

