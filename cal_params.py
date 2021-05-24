import torch

from thop import profile
from thop import clever_format

from TransInvNet.model.model import TransInvNet
from TransInvNet.model.config import get_b16_config


if __name__ == '__main__':
    model = TransInvNet(get_b16_config(), img_size=352, vis=True, pretrained=False).cuda()
    input = torch.randn(4, 3, 352, 352).cuda()
    flops, params = profile(model, inputs=(input, ))
    flops, params = clever_format([flops, params], "%.3f")
    print("Parameter Number: {}, Flops: {}".format(params, flops))