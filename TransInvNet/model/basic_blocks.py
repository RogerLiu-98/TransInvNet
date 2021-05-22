import torch
import torch.nn as nn
import torch.nn.functional as F

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


class ASPP(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels=256,
                 rates=[1, 6, 12, 18]):
        super(ASPP, self).__init__()
        self.convs = nn.ModuleList([])
        for rate in rates:
            if rate == 1:
                self.convs.append(Conv2dRelu(in_channels, out_channels, kernel_size=1, padding=0, stride=1, bias=False))
            else:
                self.convs.append(Conv2dRelu(in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate, bias=False))
        self.aspp_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv2dRelu(in_channels, out_channels, kernel_size=1, padding=0, stride=1, bias=False)
        )

    def forward(self, x):
        size = x.shape[-2:]
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res.append(F.interpolate(self.aspp_pool(x), size=size, mode='bilinear', align_corners=True))
        return torch.cat(res, dim=1)


class Ra(nn.Module):
    def __init__(self, in_channels, n_classes, scale_factor=4):
        super(Ra, self).__init__()
        self.upper_branch = nn.Sequential(
            nn.Conv2d(in_channels, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        )
        self.mid_branch = nn.Sequential(
            nn.Conv2d(in_channels, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        )
        self.lower_branch = nn.Sequential(
            nn.Conv2d(in_channels, n_classes, kernel_size=1, padding=0, stride=1, bias=True)
        )
        self.scale_factor = scale_factor

    def forward(self, x):
        up_out = self.upper_branch(x)
        mid_out = self.mid_branch(x)
        low_out = self.lower_branch(x)
        # Output from the lower branch, a simple convolution
        origin_out = F.interpolate(low_out, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)

        attn_map1 = torch.sigmoid(-1 * mid_out)
        attn_map1 = attn_map1 * up_out
        attn_map2 = torch.sigmoid(mid_out)
        attn_map2 = attn_map2 * low_out
        combined_pred = mid_out - attn_map1 + attn_map2
        # Output from the mid branch, use attention map and combine outputs from other branchs
        combined_pred = F.interpolate(combined_pred, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)

        rev_pred = -1 * up_out
        # Output from the upper branch, reverse output
        rev_pred = F.interpolate(rev_pred, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)

        final_pred = origin_out + combined_pred + rev_pred

        return final_pred
