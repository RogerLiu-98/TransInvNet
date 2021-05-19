import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from TransInvNet.model.basic_blocks import Conv2dRelu, Inv2dRelu


class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            Conv2dRelu(in_channel, out_channel, 1, padding=0, dilation=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            Conv2dRelu(in_channel, out_channel, 1, padding=0, dilation=1, stride=1),
            Conv2dRelu(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1), stride=1, dilation=1),
            Conv2dRelu(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0), stride=1, dilation=1),
            Conv2dRelu(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            Conv2dRelu(in_channel, out_channel, 1, padding=0, dilation=1, stride=1),
            Conv2dRelu(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2), stride=1, dilation=1),
            Conv2dRelu(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0), stride=1, dilation=1),
            Conv2dRelu(out_channel, out_channel, 3, padding=5, dilation=5, stride=1)
        )
        self.branch3 = nn.Sequential(
            Conv2dRelu(in_channel, out_channel, 1, stride=1, padding=0, dilation=1),
            Conv2dRelu(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3), stride=1, dilation=1),
            Conv2dRelu(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0), stride=1, dilation=1),
            Conv2dRelu(out_channel, out_channel, 3, padding=7, dilation=7, stride=1)
        )
        self.conv_cat = Conv2dRelu(4 * out_channel, out_channel, 3, padding=1, stride=1, dilation=1)
        self.conv_res = Conv2dRelu(in_channel, out_channel, 1, padding=0, stride=1, dilation=1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class DecoderBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 scale_factor=1 / 8):
        super(DecoderBlock, self).__init__()
        self.proj = Conv2dRelu(in_channels * 2, out_channels, kernel_size=1, padding=0, stride=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.downscale_factor = scale_factor
        self.in_channels = in_channels

    def forward(self, x, skip, lateral_map, is_last=False):
        x = torch.cat((x, skip), dim=1)

        if not is_last:
            next_x = self.up(self.proj(x))
        else:
            next_x = None

        ra_crop = F.interpolate(lateral_map, scale_factor=self.downscale_factor, mode='bilinear', align_corners=True)
        ra_sigmoid = -1 * torch.sigmoid(ra_crop) + 1
        ra_sigmoid = ra_sigmoid.expand(-1, self.in_channels * 2, -1, -1)
        x = ra_sigmoid.mul(x)
        return next_x, x, ra_crop


class Segmentation(nn.Module):
    def __init__(self, in_channels, n_classes=1, scale_factor=2):
        super(Segmentation, self).__init__()
        self.segmentation = nn.Sequential(
            Conv2dRelu(in_channels, in_channels // 4, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels // 4, n_classes, kernel_size=1, padding=0, stride=1)
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=scale_factor)

    def forward(self, x, ra=None):
        x = self.segmentation(x)
        if ra is not None:
            x = x + ra
        x = self.up(x)
        return x


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.config = config
        self.conv_inter = nn.Sequential(
            Conv2dRelu(config.hidden_size, config.hidden_size,
                       kernel_size=3, padding=1, stride=1),
            Conv2dRelu(config.hidden_size, config.inter_channel,
                       kernel_size=1, padding=0, stride=1, bias=True)
        )

        decoder_channels = config.decoder_channels
        in_channels = [self.config.inter_channel] + list(decoder_channels)[1:]
        out_channels = decoder_channels[1:]

        self.rfbs = nn.ModuleList([
            nn.Sequential(
                RFB_modified(in_ch, 32),
                nn.UpsamplingBilinear2d(scale_factor=factor)
            )
            for factor, in_ch in zip(config.rfb_upsample_factors, config.rfb_channels)
        ])

        self.seg_global_map = Segmentation(32 * len(self.rfbs), self.config.n_classes, self.config.scale_factors[0])

        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(in_ch, out_ch, scale_factor) for in_ch, out_ch, scale_factor in
            zip(in_channels, out_channels, self.config.downscale_factors)
        ])

        self.segmentation_blocks = nn.ModuleList([
            Segmentation(in_ch * 2, self.config.n_classes, scale_factor) for in_ch, scale_factor in
            zip(in_channels, self.config.scale_factors[1:])
        ])

    def forward(self, x, features):
        lateral_maps = []
        B, n_patch, hidden = x.size()
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))

        x = x.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)

        rfb_features = [x] + features
        for i, rfb in enumerate(self.rfbs):
            rfb_features[i] = rfb(rfb_features[i])

        global_features = torch.cat(rfb_features, dim=1)
        global_map = self.seg_global_map(global_features)
        lateral_maps.append(global_map)

        next_x = self.conv_inter(x)
        for i, (decoder_block, segmentation_block) in \
                enumerate(zip(self.decoder_blocks, self.segmentation_blocks), start=1):
            if i != len(self.decoder_blocks):
                next_x, x, ra_crop = decoder_block(next_x, features[i - 1], lateral_maps[i - 1], is_last=False)
                lateral_map = segmentation_block(x, ra_crop)
                lateral_maps.append(lateral_map)
            else:
                _, x, ra_crop = decoder_block(next_x, features[i - 1], lateral_maps[i - 1], is_last=True)
                final_map = segmentation_block(x, ra_crop)

        return final_map
