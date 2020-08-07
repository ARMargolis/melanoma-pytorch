
import torch
import torch.nn as nn
import torch.nn.functional as F


class MelanomaNet(nn.Module):
    def __init__(self, **architechture):
        super(MelanomaNet, self).__init__()
        #EXPECTS 244 BY 244 INPUT! resnet paper: https://arxiv.org/pdf/1512.03385.pdf

        channels = [3,64,64,64,128,128,256,256,512,512] #[3,64,256,512,1020,2040,1020,512,1]
        channels = list(zip(channels,channels[1:]))[1:] #first layer is not a resblock, just a conv.
        kernels = [(3,3) for block in channels]
        strides = [3 for block in channels]

        # intro block.
        # TODO don't forget to normalize the input. BatchNorm or just make sure pixel vals are scaled properly
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=channels[0][0],kernel_size=(3,3),stride=1,bias=False, padding=1)
        self.intro = nn.Sequential(
            # Don't forget regularization in the optimizer
            self.conv1,
            #nn.MaxPool2d(kernel_size=(3,3), stride=2, padding =1),
            nn.BatchNorm2d(channels[0][0])
        )
        
        residual_layers = [BasicBlock(in_channels=channels[i][0],out_channels=channels[i][1],stride=strides[i]) for i in range(len(channels))]
        self.residual_layers = nn.Sequential(*residual_layers) 
        
        #self.avg_pool = nn.AvgPool2d()

        self.block = BasicBlock(in_channels=3, out_channels=6, stride = 3)

    def forward(self, x):
        x = self.intro(x)
        print(x.clone().shape)
        x = self.residual_layers(x)
        return x



# Credit for all code below goes to https://github.com/hysts/pytorch_resnet/blob/master/resnet.py
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,  # downsample with first conv
            padding=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module(
                'conv',
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,  # downsample
                    padding=0,
                    bias=False))
            self.shortcut.add_module('bn', nn.BatchNorm2d(out_channels))  # BN

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.bn2(self.conv2(y))
        y += self.shortcut(x)
        y = F.relu(y, inplace=True)  # apply ReLU after addition
        return y


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride):
        super(BottleneckBlock, self).__init__()

        bottleneck_channels = out_channels // self.expansion

        self.conv1 = nn.Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)

        self.conv2 = nn.Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride,  # downsample with 3x3 conv
            padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)

        self.conv3 = nn.Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()  # identity
        if in_channels != out_channels:
            self.shortcut.add_module(
                'conv',
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,  # downsample
                    padding=0,
                    bias=False))
            self.shortcut.add_module('bn', nn.BatchNorm2d(out_channels))  # BN

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = F.relu(self.bn2(self.conv2(y)), inplace=True)
        y = self.bn3(self.conv3(y))  # not apply ReLU
        y += self.shortcut(x)
        y = F.relu(y, inplace=True)  # apply ReLU after addition
        return y