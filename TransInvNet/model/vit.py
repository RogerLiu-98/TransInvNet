# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math
from collections import OrderedDict
from os.path import join as pjoin

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair

import TransInvNet.model.config as configs
from TransInvNet.model.rednet import RedNet
from TransInvNet.utils.involution_cuda import involution

logger = logging.getLogger(__name__)

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


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

    def forward(self, x):
        x = self.conv(x)
        return x


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

    def forward(self, x):
        x = self.inv(x)
        return x


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


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


class IDA(nn.Module):
    def __init__(self, node_kernel, out_dim, channels, up_factors):
        super(IDA, self).__init__()
        self.channels = channels
        self.out_dim = out_dim
        for i, c in enumerate(channels):
            if c == out_dim:
                proj = nn.Identity()
            else:
                proj = Conv2dRelu(c, out_dim, kernel_size=1, stride=1, bias=False)
            f = int(up_factors[i])
            if f == 1:
                up = nn.Identity()
            else:
                up = nn.Upsample(scale_factor=f, mode='bilinear', align_corners=True)
            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)

        for i in range(1, len(channels)):
            node = Conv2dRelu(out_dim * 2, out_dim, kernel_size=node_kernel, stride=1, padding=node_kernel // 2,
                              bias=False)
            setattr(self, 'node_' + str(i), node)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.decode = nn.Conv2d(out_dim, 1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, layers):
        assert len(self.channels) == len(layers), \
            '{} vs {} layers'.format(len(self.channels), len(layers))
        layers = list(layers)
        for i, l in enumerate(layers):
            upsample = getattr(self, 'up_' + str(i))
            project = getattr(self, 'proj_' + str(i))
            layers[i] = upsample(project(l))
        x = layers[0]
        for i in range(1, len(layers)):
            node = getattr(self, 'node_' + str(i))
            x = node(torch.cat([x, layers[i]], 1))
        x = self.decode(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """

    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:  # ResNet
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False
        if self.hybrid:
            self.hybrid_model = RedNet(depth=config.rednet.depth, num_stages=config.rednet.num_stages,
                                       strides=config.rednet.strides,
                                       dilations=config.rednet.dilations,
                                       base_channels=config.rednet.base_channels,
                                       out_indices=config.rednet.out_indices)
            in_channels = self.hybrid_model.base_channels * 16
            self.projs = nn.ModuleList([nn.Sequential(
                Conv2dRelu(in_ch, in_ch // 2, kernel_size=1, padding=0, stride=1),
                nn.UpsamplingBilinear2d(scale_factor=2))
                for in_ch in config.rednet.out_channels
            ])
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))
        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        if self.hybrid:
            features = self.hybrid_model(x)
            for i, proj in enumerate(self.projs):
                features[i] = proj(features[i])
                x = features[0]
        else:
            features = None
        x = self.patch_embeddings(x)  # (B, hidden, n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(start_dim=2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size,
                                                                                   self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size,
                                                                                   self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size,
                                                                                   self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, attn_weights, features


class MSCAM(nn.Module):

    def __init__(self, in_channels, r=4):
        super(MSCAM, self).__init__()
        inter_channels = in_channels // r
        self.local_att = nn.Sequential(
            Conv2dRelu(in_channels, inter_channels, kernel_size=1, padding=0, stride=1),
            nn.Conv2d(inter_channels, in_channels, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(in_channels)
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv2dRelu(in_channels, inter_channels, kernel_size=1, padding=0, stride=1, bias=True),
            nn.Conv2d(inter_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = torch.sigmoid(xlg)
        return wei


class iAFF(nn.Module):

    def __init__(self, in_channels, out_channels, r=4):
        super(iAFF, self).__init__()
        self.ms_cam1 = MSCAM(in_channels, r)
        self.ms_cam2 = MSCAM(in_channels, r)
        self.proj = nn.Sequential(
            Inv2dRelu(in_channels, kernel_size=7, stride=1),
            Conv2dRelu(in_channels, out_channels, kernel_size=1, padding=0, stride=1, bias=True),
        )

    def forward(self, x, y):
        x1 = x + y
        x1l = self.ms_cam1(x1)
        x1r = 1 - x1l
        x2 = x * x1l + y * x1r

        x2l = self.ms_cam2(x2)
        x2r = 1 - x2l
        z = x * x2l + y * x2r

        z = self.proj(z)
        return z


class DecoderBlock(nn.Module):
    def __init__(self, scale_factor):
        super(DecoderBlock, self).__init__()
        # self.iaff = iAFF(in_channels, out_channels, r=4)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.downscale_factor = scale_factor

    def forward(self, x, lateral_map):
        x = self.up(x)
        ra_crop = F.interpolate(lateral_map, scale_factor=self.downscale_factor, mode='bilinear', align_corners=True)
        ra_sigmoid = -1 * torch.sigmoid(ra_crop) + 1
        ra_sigmoid = ra_sigmoid.expand(-1, x.size()[1], -1, -1)
        x = ra_sigmoid.mul(x)
        return x, ra_crop


class Segmentation(nn.Module):
    def __init__(self, in_channels, n_classes=1, scale_factor=2, r=4):
        super(Segmentation, self).__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=scale_factor)
        self.segmentation = nn.Sequential(
            Inv2dRelu(in_channels, kernel_size=7, stride=1),
            nn.Conv2d(in_channels, n_classes, kernel_size=1, padding=0, stride=1)
        )

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
            Inv2dRelu(config.hidden_size, kernel_size=7, stride=1),
            Conv2dRelu(config.hidden_size, config.inter_channel,
                       kernel_size=1, padding=0, stride=1, bias=True),
        )

        decoder_channels = config.decoder_channels
        out_channels = decoder_channels[1:]

        self.proj = Conv2dRelu(config.hidden_size, 32, kernel_size=1, padding=0, stride=1)
        self.rfbs = nn.ModuleList([
            RFB_modified(in_ch, 32)
            for in_ch in config.rfb_channels
        ])

        self.aggregation = nn.Sequential(
            IDA(node_kernel=3, out_dim=32, channels=[32, 32, 32, 32], up_factors=[4, 4, 2, 1]),
            nn.UpsamplingBilinear2d(scale_factor=self.config.scale_factors[0])
        )

        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(scale_factor) for scale_factor in self.config.downscale_factors
        ])

        self.segmentation_blocks = nn.ModuleList([
            Segmentation(out_ch, self.config.n_classes, scale_factor) for out_ch, scale_factor in
            zip(out_channels, self.config.scale_factors[1:])
        ])

    def forward(self, x, features):
        lateral_maps = []
        B, n_patch, hidden = x.size()
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))

        x = x.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        rfb_features = [x] + features
        rfb_features[0] = self.proj(rfb_features[0])
        for i, rfb in enumerate(self.rfbs, start=1):
            rfb_features[i] = rfb(rfb_features[i])
        global_map = self.aggregation(rfb_features)
        lateral_maps.append(global_map)

        for i, (decoder_block, segmentation_block) in \
                enumerate(zip(self.decoder_blocks, self.segmentation_blocks)):
            x, ra_crop = decoder_block(features[i], lateral_maps[i])
            laterap_map = segmentation_block(x, ra_crop)
            lateral_maps.append(laterap_map)
        return lateral_maps


class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis)
        self.config = config
        self.Decoder = Decoder(config)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden))
        lateral_maps = self.Decoder(x, features)
        return lateral_maps[0], lateral_maps[1], lateral_maps[2], lateral_maps[3]

    def load_from(self, weights):
        with torch.no_grad():

            res_weight = weights
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
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'testing': configs.get_testing(),
}

if __name__ == '__main__':
    config = configs.get_r50_b16_config()
    model = VisionTransformer(config, img_size=256, vis=True).cuda()
    model.load_from(np.load(config.pretrained_path))
    im = torch.randn((4, 3, 256, 256)).cuda()
    x4, x3, x2, x1 = model(im)
    print(x1.size(), x2.size(), x3.size(), x4.size())
    # print(model)