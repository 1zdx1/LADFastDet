import math

import torch
import torch.nn as nn



#   BasicConv

# class BasicConv(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1):
#         super(BasicConv, self).__init__()
#
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.activation = nn.LeakyReLU(0.1)
#
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         x = self.activation(x)
#         return x

class BasicConv_hardswish(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicConv_hardswish, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.Hardswish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

# DepthwiseConv
# class DepthwiseConv(nn.Module):
#     def __init__(self, in_channels,out_channels, kernel_size, stride=1):
#         super(DepthwiseConv, self).__init__()
#
#         self.depthwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, groups=in_channels, bias=False)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.activation = nn.ReLU()
#
#     def forward(self, x):
#         x = self.depthwise_conv(x)
#         x = self.bn(x)
#         x = self.activation(x)
#         return x

class DepthwiseConv_hardswish(nn.Module):
    def __init__(self, in_channels,out_channels, kernel_size, stride=1):
        super(DepthwiseConv_hardswish, self).__init__()

        self.depthwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, groups=in_channels, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.Hardswish()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

# InvResblock_body

class InvResblock_body_1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InvResblock_body_1, self).__init__()
        self.out_channels = out_channels

        self.conv1 = DepthwiseConv_hardswish(in_channels, out_channels, 3)
        self.conv2 = BasicConv_hardswish(out_channels, out_channels * 2, 1)
        self.conv3 = DepthwiseConv_hardswish(out_channels * 2, out_channels * 2, 3)
        self.conv4 = BasicConv_hardswish(out_channels *4, out_channels, 1)
        self.maxpool = nn.MaxPool2d([2,2],[2,2])

    def forward(self, x):
        x = self.conv1(x)
        route = x
        c = self.out_channels
        x = self.conv2(x)
        route1 = x
        x = self.conv3(x)
        x = self.conv4(x)
        x = torch.cat([route, x], dim = 1)
        x = self.maxpool(x)
        return x

class InvResblock_body_2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InvResblock_body_2, self).__init__()
        self.out_channels = out_channels

        self.conv1 = DepthwiseConv_hardswish(in_channels, out_channels, 3)
        self.conv2 = BasicConv_hardswish(out_channels, out_channels * 2, 1)
        self.conv3 = DepthwiseConv_hardswish(out_channels * 2, out_channels * 2, 3)
        self.conv4 = BasicConv_hardswish(out_channels * 2, out_channels, 1)
        self.maxpool = nn.MaxPool2d([2, 2], [2, 2])

    def forward(self, x):
        x = self.conv1(x)
        route = x
        c = self.out_channels
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = torch.cat([route, x], dim=1)
        x = self.maxpool(x)
        return x

class Lightweight_InvResblock(nn.Module):
    def __init__(self):
        super(Lightweight_InvResblock, self).__init__()
        self.conv1 = BasicConv_hardswish(3, 32, kernel_size=3, stride=2)
        self.conv2 = BasicConv_hardswish(32, 32, kernel_size=3, stride=2)
        self.invresblock_body1 = InvResblock_body_1(32, 32)
        self.invresblock_body2 = InvResblock_body_2(64, 64)
        self.invresblock_body3 = InvResblock_body_2(128, 128)
        self.num_features = 1

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.invresblock_body1(x)
        x = self.invresblock_body2(x)
        x = self.invresblock_body3(x)
        return x

def invresblock(pretrained, **kwargs):
    model = Lightweight_InvResblock()
    return model

