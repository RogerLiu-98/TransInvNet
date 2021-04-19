import argparse
import pathlib

import albumentations as A
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from albumentations.pytorch.transforms import ToTensorV2
from tqdm import tqdm

from TransInvNet.model.vit import VisionTransformer, CONFIGS


class Metrics:
    def __init__(self, pred, mask):
        self.pred, self.mask = pred.flatten(), mask.flatten()
        self.intersection = np.sum(self.pred * self.mask)
        self.mask_sum = np.sum(np.abs(self.pred)) + np.sum(np.abs(self.mask))
        self.union = self.mask_sum - self.intersection
        self.abs_error = np.abs(self.pred - self.mask)

    def calculate_iou(self):
        return (self.intersection + 1e-8) / (self.union + 1e-8)

    def calculate_mae(self):
        return np.mean(self.abs_error)

    def calculate_dice(self):
        return (self.intersection * 2 + 1e-8) / (self.mask_sum + 1e-8)

    def _S_object(self):
        fg = np.where(self.mask == 0, np.zeros_like(self.pred), self.pred)
        bg = np.where(self.mask == 1, np.zeros_like(self.pred), 1 - self.pred)
        o_fg = self._object(fg, self.mask)
        o_bg = self._object(bg, 1 - self.mask)
        u = np.mean(self.mask)
        Q = u * o_fg + (1 - u) * o_bg
        return Q

    def _object(self, pred, mask):
        temp = pred[mask == 1]
        x = np.mean(temp)
        sigma_x = np.std(temp)
        score = 2.0 * x / (x * x + 1.0 + sigma_x + 1e-8)
        return score

    def calculate_s_measure(self):
        mask_mean = np.mean(self.mask)
        pred_mean = np.mean(self.pred)
        if mask_mean == 0:
            Q = 1 - pred_mean
        elif mask_mean == 1:
            Q = pred_mean
        else:
            Q = np.maximum(0.5 * self._S_object() + (1 - 0.5) * self._S_object(), 0)
        return Q

    def _eval_e(self, num=255):
        score = np.zeros(num)
        for i in range(num):
            fm = self.pred - np.mean(self.pred)
            gt = self.mask - np.mean(self.mask)
            align_matrix = 2 * gt * fm / (gt * gt + fm * fm + 1e-8)
            enhanced = ((align_matrix + 1) * (align_matrix + 1)) / 4
            score[i] = np.sum(enhanced) / (len(self.mask) - 1 + 1e-8)
        return np.max(score)

    def calculate_e_measure(self):
        max_e = self._eval_e(255)
        return max_e


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', type=int,
                        default=352, help='training dataset size')
    parser.add_argument('--weight_path', type=str,
                        default='outputs/exp04190514/train/TransInvNet-best.pth', help='path to the trained weight')
    parser.add_argument('--test_path', type=str,
                        default='datasets/polyp-dataset/kvasir/test', help='path to test dataset')
    opt = parser.parse_args()

    cfg = CONFIGS['R50-ViT-B_16']
    model = VisionTransformer(cfg, opt.img_size, vis=True).cuda()
    model.load_state_dict(torch.load(opt.weight_path))
    model.eval()

    test_images_path = [i for i in (pathlib.Path(opt.test_path) / 'images').iterdir()]
    test_masks_path = [i for i in (pathlib.Path(opt.test_path) / 'masks').iterdir()]

    trans = A.Compose([
        A.Normalize(mean=[0.497, 0.302, 0.216],
                    std=[0.320, 0.217, 0.173]),
        ToTensorV2(),
    ])

    Miou, Amae, Mdice, Smeasure, Emeasure = [], [], [], [], []
    with torch.no_grad():
        tbar = tqdm(zip(test_images_path, test_masks_path), desc='\r', total=len(test_images_path))
        for img, gt in tbar:
            img = Image.open(img).convert('RGB').resize((opt.img_size, opt.img_size))
            img = np.array(img)
            img = trans(image=img)['image'][None]
            img = img.cuda()
            gt = np.asarray(Image.open(gt).convert('1')).astype(np.float)
            _, _, _, pred = model(img)
            pred = F.interpolate(pred, size=gt.shape, mode='bilinear', align_corners=True)
            result = pred.sigmoid().cpu().numpy().squeeze()
            result = (result - result.min()) / (result.max() - result.min() + 1e-8)

            metrics = Metrics(result, gt)
            iou = metrics.calculate_iou()
            mae = metrics.calculate_mae()
            dice = metrics.calculate_dice()
            s_measure = metrics.calculate_s_measure()
            e_measure = metrics.calculate_e_measure()
            Miou.append(iou)
            Amae.append(mae)
            Mdice.append(dice)
            Smeasure.append(s_measure)
            Emeasure.append(e_measure)

            tbar.set_description('Mean IOU: {:.4f}, Average MAE: {:.4f}, Mean DICE: {:.4f}, '
                                 'S Measure {:.4f}, E Measure {:.4f}'
                                 .format(np.mean(Miou), np.mean(Amae), np.mean(Mdice), np.mean(Smeasure),
                                         np.mean(Emeasure)))
