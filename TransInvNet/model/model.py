# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

import numpy as np
from scipy import ndimage
from collections import OrderedDict

from TransInvNet.model.vit.vit import Transformer, logger
from TransInvNet.model.decoder.decoder import Decoder
from TransInvNet.utils.utils import np2th
import TransInvNet.model.config as configs


class TransInvNet(nn.Module):
    def __init__(self, config, img_size=352, zero_head=False, vis=True):
        super(TransInvNet, self).__init__()
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.config = config

        self.transformer = Transformer(config, img_size, vis)
        self.decoder = Decoder(config)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden))
        out = self.decoder(x, features)
        return out

    def load_from(self, weights):
        with torch.no_grad():

            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1] - 1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            # Encoder whole
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                hybrid_state_dict = torch.load(self.config.rednet.pretrained_path)['state_dict']
                hybrid_state_dict = OrderedDict({'.'.join(k.split('.')[1:]): v for k, v in hybrid_state_dict.items()
                                                 if k.split('.')[0] == 'backbone'})
                self.transformer.embeddings.hybrid_model.load_state_dict(hybrid_state_dict)


CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
}

if __name__ == '__main__':
    model = TransInvNet(CONFIGS['R50-ViT-B_16'], img_size=256, vis=True).cuda()
    x = torch.randn((4, 3, 256, 256)).cuda()
    out = model(x)[-1]
    print(out.size())
