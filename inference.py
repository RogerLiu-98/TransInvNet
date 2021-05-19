import argparse
import os
import pathlib

import albumentations as A
import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from albumentations.pytorch.transforms import ToTensorV2
from tqdm import tqdm

from TransInvNet.model.model import TransInvNet, CONFIGS

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', type=int,
                        default=352, help='training dataset size')
    parser.add_argument('--weight_path', type=str,
                        default='outputs/exp04190514/train/TransInvNet-best.pth', help='path to the trained weight')
    parser.add_argument('--test_path', type=str,
                        default='datasets/polyp-dataset/kvasir/test', help='path to test dataset')
    parser.add_argument('--output_path', type=str,
                        default='outputs/exp04190514/inference', help='path to output path')
    parser.add_argument('--threshold', type=int,
                        default=0.5, help='sigmoid threshold')
    opt = parser.parse_args()

    cfg = CONFIGS['R50-ViT-B_16']
    model = TransInvNet(cfg, opt.img_size, vis=True).cuda()
    model.load_state_dict(torch.load(opt.weight_path))
    model.eval()

    test_images_path = [i for i in (pathlib.Path(opt.test_path) / 'images').iterdir()]

    trans = A.Compose([
        A.Normalize(mean=[0.497, 0.302, 0.216],
                    std=[0.320, 0.217, 0.173]),
        ToTensorV2(),
    ])

    os.makedirs(opt.output_path, exist_ok=True)

    with torch.no_grad():
        tbar = tqdm(test_images_path, desc='\r', )
        for img in tbar:
            file_name = img.name
            im = Image.open(img).convert('RGB')
            w, h = im.size
            im = im.resize((opt.img_size, opt.img_size))
            img = np.array(im)
            img = trans(image=img)['image'][None]
            img = img.cuda()
            _, _, pred, _ = model(img)
            pred = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=True)
            result = pred.sigmoid().cpu().numpy().squeeze()
            result[result >= 0.5] = 1
            result[result < 0.5] = 0
            pred_im = Image.fromarray(np.uint8(result * 255), 'L')
            pred_im.save(pathlib.Path(opt.output_path) / file_name)
