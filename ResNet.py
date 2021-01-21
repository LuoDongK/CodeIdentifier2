from abc import ABC

import torch
from torch import nn

torch.manual_seed(123)


class BasicBlock(nn.Module, ABC):
    def __init__(self, in_channel, out_channel, kernel, strides, padding=0, bias=False, relu=True):
        super(BasicBlock, self).__init__()
        self.relu = relu
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel, stride=strides, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channel),
        )

    def forward(self, inputs):
        out = self.conv(inputs)
        if self.relu:
            out = nn.ReLU(out)
            return out
        else:
            return out


def build_resblock(in_channel, out_channel, blocks, stride=1):
    resblock = nn.Sequential()
    resblock.add_module('Resblock0', ResBlock(in_channel, out_channel, stride))
    for index in range(1, blocks):
        resblock.add_module('Resblock{}'.format(str(index)), ResBlock(out_channel, out_channel, 1))
    return resblock


class ResBlock(nn.Module, ABC):
    def __init__(self, in_channel, out_channel, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = BasicBlock(in_channel, out_channel, 3, stride, 1)
        self.conv2 = BasicBlock(out_channel, out_channel, 3, 1, 1, relu=False)

        self.downsample = nn.Sequential()
        if in_channel != out_channel or stride != 1:
            self.downsample = BasicBlock(in_channel, out_channel, 1, stride, relu=False)

    def forward(self, inputs):
        out = self.conv1(inputs)
        out = self.conv2(out)
        out = self.downsample(inputs) + out
        out = nn.ReLU(out)
        return out


class ResNet(nn.Module, ABC):
    def __init__(self, layer_dim, num_classes):
        super(ResNet, self).__init__()
        self.conv = nn.Sequential(
            BasicBlock(3, 64, 7, 3),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        layer1 = build_resblock(64, 64, layer_dim[0])
        layer2 = build_resblock(64, 128, layer_dim[1], stride=2)
        layer3 = build_resblock(128, 256, layer_dim[2], stride=2)
        layer4 = build_resblock(256, 512, layer_dim[3], stride=2)

        self.blocks = nn.Sequential(
            layer1,
            layer2,
            layer3,
            layer4
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512, num_classes)

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.blocks(out)
        out = self.global_avg_pool(out)
        return out


def resnet18():
    return ResNet([2, 2, 2, 2])


def resnet34():
    return ResNet([3, 4, 6, 3])


if __name__ == '__main__':
    x = torch.rand((1, 3, 100, 200))
    model = resnet18()
    model(x)
