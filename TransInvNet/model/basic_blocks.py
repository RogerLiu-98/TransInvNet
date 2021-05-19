import torch
import torch.nn as nn

from TransInvNet.utils.involution_cuda import involution


class Inv2dRelu(nn.Module):
    def __init__(self,
                 channels,
                 kernel_size,
                 stride):
        super(Inv2dRelu, self).__init__()
        self.inv = nn.Sequential(
            involution(channels, kernel_size, stride),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        self.init_weights()

    def forward(self, x):
        x = self.inv(x)
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Conv2dRelu(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding=0,
                 stride=1,
                 dilation=1,
                 bias=True):
        super(Conv2dRelu, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size, padding=padding,
                      stride=stride, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.init_weights()

    def forward(self, x):
        x = self.conv(x)
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
