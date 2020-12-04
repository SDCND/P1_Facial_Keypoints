## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting

        # 2 input image channel (grayscale), 64 output channels/feature maps, 5x5 square convolution kernel
        self.conv2 = nn.Conv2d(32, 64, 5)
        
        # 3 input image channel (grayscale), 128 output channels/feature maps, 3x3 square convolution kernel
        self.conv3 = nn.Conv2d(64, 128, 3)

        # 4 input image channel (grayscale), 256 output channels/feature maps, 3x3 square convolution kernel
        self.conv4 = nn.Conv2d(128, 256, 3)
        
        # 1st linear layer
        self.linear1 = nn.Linear(30976, 272)  # 30976 = 11 * 11 * 256
        
        # 2nd linear layer
        self.linear2 = nn.Linear(272, 136)

         # stepwise 2d batch normalizations
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.batch_norm4 = nn.BatchNorm2d(256)
        
        # 2d pool with kernel_size=2, stride=2
        self.pool1 = nn.MaxPool2d(2, 2)

        # 2d dropout with p = 0.3
        self.drop1 = nn.Dropout2d(0.3)

        # dropout with p = 0.3
        self.drop2 = nn.Dropout(0.3)

        # 2d adatpive average pool
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        m = nn.LeakyReLU(0.05)

        x = self.pool1(m(self.batch_norm1(self.conv1(x))))
        x = self.drop1(x)
        x = self.pool1(m(self.batch_norm2(self.conv2(x))))
        x = self.drop1(x)
        x = self.pool1(m(self.batch_norm3(self.conv3(x))))
        x = self.drop1(x)
        x = self.pool1(m(self.batch_norm4(self.conv4(x))))

        # prepare for linear layer (flatten)
        x = x.view(x.size(0), -1)

        x = self.linear1(x)
        x = self.drop2(x)
        x = self.linear2(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
