import ml_collections


def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1

    config.classifier = 'seg'
    config.representation_size = None
    config.resnet_pretrained_path = None
    config.pretrained_path = '../model/vit_checkpoint/imagenet21k/ViT-B_16.npz'
    config.patch_size = 16

    config.decoder_channels = (256, 128, 64, 16)
    config.n_classes = 1
    return config


def get_r50_b16_config():
    """Returns the Rednet50 + ViT-B/16 configuration."""
    config = get_b16_config()
    config.patches.grid = (16, 16)

    config.rednet = ml_collections.ConfigDict()
    config.rednet.depth = 50
    config.rednet.num_stages = 4
    config.rednet.strides = (1, 2, 2, 2)
    config.rednet.dilations = (1, 1, 1, 1)
    config.rednet.base_channels = 64
    config.rednet.out_indices = (1, 2, 3)
    config.rednet.out_channels = [2048, 1024, 512]
    config.rednet.pretrained_path = ('TransInvNet/lib/rednet50-1c7a7c5d.pth')

    config.classifier = 'seg'
    config.pretrained_path = 'TransInvNet/lib/imagenet21k+imagenet2012_R50+ViT-B_16.npz'
    config.inter_channel = 1024
    config.decoder_channels = (768, 512, 256, 64)
    config.rfb_channels = [768, 1024, 512, 256]
    config.rfb_upsample_factors = [4, 4, 2, 1]
    config.scale_factors = [4, 16, 8, 4]
    config.downscale_factors = [1 / 16, 1 / 8, 1 / 4]
    config.n_classes = 1

    return config
