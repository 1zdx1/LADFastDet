import torch
import torch.nn as nn

from nets.Lightweight_InvResblock import invresblock

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

# class DepthwiseConv(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1):
#         super(DepthwiseConv, self).__init__()
#
#         self.depthwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, groups=in_channels, bias=False)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.activation = nn.LeakyReLU(0.1)
#
#     def forward(self, x):
#         x = self.depthwise_conv(x)
#         x = self.bn(x)
#         x = self.activation(x)
#         return x

class DepthwiseConv_hardswish(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(DepthwiseConv_hardswish, self).__init__()

        self.depthwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, groups=in_channels, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.Hardswish()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class SeparableConv_hardswish(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(SeparableConv_hardswish, self).__init__()

        self.depthwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, groups=in_channels, bias=False)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.Hardswish()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class MultiScale_FFN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScale_FFN, self).__init__()
        self.out_channels = out_channels
        self.conv1 = BasicConv_hardswish(in_channels, out_channels, 1)
        self.conv2 = DepthwiseConv_hardswish(in_channels, out_channels, 3)
        self.conv3 = DepthwiseConv_hardswish(in_channels, out_channels, 5)
        self.conv4 = BasicConv_hardswish(out_channels *2, out_channels, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x = torch.cat([x2, x3], dim=1)
        feat = x
        x = self.conv4(x)
        x = torch.cat([x1, x], dim=1)
        return x, feat

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            BasicConv_hardswish(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
    def forward(self, x,):
        x = self.upsample(x)
        return x

def head(filters_list, in_filters):
    m = nn.Sequential(
        SeparableConv_hardswish(in_filters, filters_list[0], 3),
        nn.Conv2d(filters_list[0], filters_list[1], 1),
    )
    return m

class LADFastDet(nn.Module):
    def __init__(self, anchors_mask, num_classes, phi=0, pretrained=False):
        super(LADFastDet, self).__init__()
        self.phi      = phi
        self.backbone = invresblock(pretrained)
        self.MFF      = MultiScale_FFN(256, 256)
        self.head1    = head([512, len(anchors_mask[0]) * (5 + num_classes)], 512)
        self.upsample = Upsample(512,256)
        self.head01   = head([256, len(anchors_mask[1]) * (5 + num_classes)], 256)

    def forward(self, x):
        x = self.backbone(x)
        feat1, feat2 = self.MFF(x)
        Upsample = self.upsample(feat1)
        out0 = self.head0(Upsample)
        out1= self.head1(feat2)

        return out0, out1
