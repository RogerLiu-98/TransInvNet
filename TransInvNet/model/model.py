# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

import numpy as np

from TransInvNet.model.vit.vit import Transformer
from TransInvNet.model.backbone.rednet import RedNet
from TransInvNet.model.decoder.decoder import Decoder
import TransInvNet.model.config as configs


class TransInvNet(nn.Module):
    def __init__(self, config, img_size=352, vis=True, pretrained=True):
        super(TransInvNet, self).__init__()
        self.rednet = RedNet(depth=config.rednet.depth,
                             num_stages=config.rednet.stages,
                             strides=config.rednet.strides,
                             dilations=config.rednet.dilations,
                             out_indices=config.rednet.out_indices)
        self.transformer = Transformer(config, img_size, vis=vis)
        self.decoder = Decoder(config)
        if pretrained is True:
            self.rednet.init_weights(config.rednet.pretrained_path)
            self.transformer.load_from(np.load(config.pretrained_path))
        else:
            self.rednet.init_weights(pretrained=None)

    def forward(self, x):
        rednet_outs = self.rednet(x)
        transformer_out = self.transformer(x)
        output_map = self.decoder(rednet_outs, transformer_out)
        return output_map


CONFIGS = {
    'ViT-B_8': configs.get_b8_config(),
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
}

if __name__ == '__main__':
    model = TransInvNet(CONFIGS['ViT-B_8'], img_size=256, vis=True, pretrained=False).cuda()
    x = torch.randn((4, 3, 256, 256)).cuda()
    out = model(x)
    print(out.size())
