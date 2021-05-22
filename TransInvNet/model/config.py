import ml_collections


def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    # -------- Configs for Transformer --------
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.transformer.inter_channel = 1024

    config.classifier = 'seg'
    config.pretrained_path = 'TransInvNet/lib/imagenet21k_ViT-B_16.npz'
    config.patch_size = 16
    config.n_classes = 1

    # -------- Configs for RedNet --------
    config.rednet = ml_collections.ConfigDict()
    config.rednet.depth = 50
    config.rednet.stages = 4
    config.rednet.strides = (1, 2, 2, 2)
    config.rednet.dilations = (1, 1, 1, 1)
    config.rednet.out_indices = (0, 1, 2, 3)
    config.rednet.out_dimensions = (2048, 1024, 512, 256)
    config.rednet.aspp_rates = [1, 6, 12]

    config.rednet.pretrained_path = 'TransInvNet/lib/rednet50.pth'

    # -------- Configs for Decoder --------
    config.decoder = ml_collections.ConfigDict()
    config.decoder.segmentation_channels = 256
    config.decoder.up_scale_factors = 2

    return config


def get_b32_config():
    """Returns the ViT-B/32 configuration."""
    config = get_b16_config()
    config.patches.size = (32, 32)
    config.pretrained_path = 'TransInvNet/lib/imagenet21k_ViT-B_32.npz'
    return config


def get_b8_config():
    """Returns the ViT-B/8 configuration."""
    config = get_b16_config()
    config.patches.size = (8, 8)
    config.pretrained_path = 'TransInvNet/lib/imagenet21k_ViT-B_8.npz'
    return config


def get_l16_config():
    """Returns the ViT-L/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 1024
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 4096
    config.transformer.num_heads = 16
    config.transformer.num_layers = 24
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.transformer.inter_channel = None

    config.classifier = 'seg'
    config.pretrained_path = 'TransInvNet/lib/imagenet21k_ViT-L_16.npz'
    config.n_classes = 1

    # -------- Configs for RedNet --------
    config.rednet = ml_collections.ConfigDict()
    config.rednet.depth = 50
    config.rednet.stages = 4
    config.rednet.strides = (1, 2, 2, 2)
    config.rednet.dilations = (1, 1, 1, 1)
    config.rednet.out_indices = (0, 1, 2, 3)
    config.rednet.out_dimensions = (2048, 1024, 512, 256)
    config.rednet.aspp_rates = [1, 6, 12]

    config.rednet.pretrained_path = 'TransInvNet/lib/rednet50.pth'

    # -------- Configs for Decoder --------
    config.decoder = ml_collections.ConfigDict()
    config.decoder.segmentation_channels = 256
    config.decoder.up_scale_factors = 2

    return config


def get_l32_config():
    """Returns the ViT-L/32 configuration."""
    config = get_l16_config()
    config.patches.size = (32, 32)
    config.pretrained_path = 'TransInvNet/lib/imagenet21k_ViT-L_32.npz'
    return config