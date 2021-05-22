import torch
import torch.nn as nn
import torch.nn.functional as F

from TransInvNet.model.basic_blocks import Conv2dRelu, Inv2dRelu, ASPP, Ra


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.aspp = ASPP(config.rednet.out_dimensions[0], rates=config.rednet.aspp_rates)
        if config.transformer.inter_channel is not None:
            self.conv_inter = Conv2dRelu(config.hidden_size, config.transformer.inter_channel, kernel_size=1, padding=0,
                                         stride=1, bias=True)
        else:
            self.conv_inter = nn.Identity()

        self.up_conv1 = nn.Sequential(
            Conv2dRelu(config.rednet.out_dimensions[0] * 2, config.rednet.out_dimensions[1], kernel_size=1, padding=0,
                       stride=1),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )

        self.up_conv2 = nn.Sequential(
            Conv2dRelu(config.rednet.out_dimensions[1] * 2, config.rednet.out_dimensions[2], kernel_size=1, padding=0,
                       stride=1),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )

        self.up_conv3 = nn.Sequential(
            Conv2dRelu(config.rednet.out_dimensions[2] * 2, config.rednet.out_dimensions[3], kernel_size=1, padding=0,
                       stride=1),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )

        self.up_conv4 = nn.Sequential(
            Conv2dRelu(config.rednet.out_dimensions[3] * 2, config.rednet.out_dimensions[3], kernel_size=1, padding=0,
                       stride=1),
            Inv2dRelu(config.rednet.out_dimensions[3], kernel_size=7, stride=1),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )

        self.ra_seg = Ra(in_channels=config.decoder.segmentation_channels, n_classes=config.n_classes,
                         scale_factor=config.decoder.up_scale_factors)

    def forward(self, rednet_outs, transformer_out):
        out1 = self.aspp(rednet_outs[0])
        out2 = self.conv_inter(transformer_out)

        out2 = F.interpolate(out2, out1.size()[-2:], mode='bilinear',
                             align_corners=True)  # Resize output from two branches

        cat = torch.cat((out1, out2), dim=1)

        branch_cat1 = torch.cat((rednet_outs[0], cat), dim=1)
        branch_cat1 = self.up_conv1(branch_cat1)

        branch_cat2 = torch.cat((rednet_outs[1], branch_cat1), dim=1)
        branch_cat2 = self.up_conv2(branch_cat2)

        branch_cat3 = torch.cat((rednet_outs[2], branch_cat2), dim=1)
        branch_cat3 = self.up_conv3(branch_cat3)

        branch_cat4 = torch.cat((rednet_outs[3], branch_cat3), dim=1)
        branch_cat4 = self.up_conv4(branch_cat4)
        lateral_map = self.ra_seg(branch_cat4)

        return lateral_map
