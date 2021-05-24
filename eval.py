import argparse
import pathlib

import torch
import torch.nn.functional as F

import numpy as np
from PIL import Image
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from TransInvNet.model.model import TransInvNet, CONFIGS


class Metrics:
    def __init__(self, pred, mask):
        self.h, self.w = mask.shape[0], mask.shape[1]
        self.origin_pred, self.origin_mask = pred, mask
        self.pred, self.mask = pred.flatten(), mask.flatten()
        self.intersection = np.sum(self.pred * self.mask)
        self.mask_sum = np.sum(np.abs(self.pred)) + np.sum(np.abs(self.mask))
        self.union = self.mask_sum - self.intersection
        self.abs_error = np.abs(self.pred - self.mask)

    def calculate_iou(self):
        return (self.intersection + 1e-20) / (self.union + 1e-20)

    def calculate_mae(self):
        return np.mean(self.abs_error)

    def calculate_dice(self):
        return (self.intersection * 2 + 1e-20) / (self.mask_sum + 1e-20)

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
        score = 2.0 * x / (x * x + 1.0 + sigma_x + 1e-20)
        return score

    def _S_region(self):
        X, Y = self._centroid()
        gt1, gt2, gt3, gt4, w1, w2, w3, w4 = self._divideGT(X, Y)
        p1, p2, p3, p4 = self._dividePrediction(X, Y)
        Q1 = self._ssim(p1, gt1)
        Q2 = self._ssim(p2, gt2)
        Q3 = self._ssim(p3, gt3)
        Q4 = self._ssim(p4, gt4)
        Q = w1 * Q1 + w2 * Q2 + w3 * Q3 + w4 * Q4
        return Q

    def _centroid(self):
        if self.mask.sum() == 0:
            X = np.eye(1) * round(self.w / 2)
            Y = np.eye(1) * round(self.h / 2)
        else:
            total = self.mask.sum()
            i = np.arange(0, self.w)
            j = np.arange(0, self.h)
            X = round((self.origin_mask.sum(axis=0) * i).sum() / total)
            Y = round((self.origin_mask.sum(axis=1) * j).sum() / total)
        return X, Y

    def _divideGT(self, X, Y):
        area = self.h * self.w
        LT = self.origin_mask[:Y, :X]
        RT = self.origin_mask[:Y, X:self.w]
        LB = self.origin_mask[Y:self.h, :X]
        RB = self.origin_mask[Y:self.h, X:self.w]

        w1 = (X * Y) / area
        w2 = (self.w - X) * Y / area
        w3 = X * (self.h - Y) / area
        w4 = 1 - w1 - w2 - w3
        return LT, RT, LB, RB, w1, w2, w3, w4

    def _dividePrediction(self, X, Y):
        LT = self.origin_pred[:Y, :X]
        RT = self.origin_pred[:Y, X:self.w]
        LB = self.origin_pred[Y:self.h, :X]
        RB = self.origin_pred[Y:self.h, X:self.w]
        return LT, RT, LB, RB

    def _ssim(self, pred, mask):
        N = self.h * self.w
        x = pred.mean()
        y = mask.mean()
        sigma_x2 = ((pred - x) * (pred - x)).sum() / (N - 1 + 1e-20)
        sigma_y2 = ((mask - y) * (mask - y)).sum() / (N - 1 + 1e-20)
        sigma_xy = ((pred - x) * (mask - y)).sum() / (N - 1 + 1e-20)

        alpha = 4 * x * y * sigma_xy
        beta = (x * x + y * y) * (sigma_x2 + sigma_y2)

        if alpha != 0:
            Q = alpha / (beta + 1e-20)
        elif alpha == 0 and beta == 0:
            Q = 1.0
        else:
            Q = 0
        return Q

    def calculate_s_measure(self):
        mask_mean = np.mean(self.mask)
        pred_mean = np.mean(self.pred)
        if mask_mean == 0:
            Q = 1 - pred_mean
        elif mask_mean == 1:
            Q = pred_mean
        else:
            Q = np.maximum(0.5 * self._S_object() + (1 - 0.5) * self._S_region(), 0)
        return Q

    def _eval_e(self, num=255):
        score = np.zeros(num)
        for i in range(num):
            fm = self.pred - np.mean(self.pred)
            gt = self.mask - np.mean(self.mask)
            align_matrix = 2 * gt * fm / (gt * gt + fm * fm + 1e-20)
            enhanced = ((align_matrix + 1) * (align_matrix + 1)) / 4
            score[i] = np.sum(enhanced) / (len(self.mask) - 1 + 1e-20)
        return np.max(score)

    def calculate_e_measure(self):
        max_e = self._eval_e(255)
        return max_e


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', type=int,
                        default=352, help='training dataset size')
    parser.add_argument('--weight_path', type=str,
                        default='outputs/exp05240731/train/TransInvNet-best.pth', help='path to the trained weight')
    parser.add_argument('--test_path', type=str,
                        default='datasets/polyp-dataset/'
                                'TestDataset/CVC-ColonDB', help='path to test dataset')
    opt = parser.parse_args()

    cfg = CONFIGS['ViT-B_16']
    model = TransInvNet(cfg, opt.img_size, vis=True).cuda()
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
            pred = model(img)
            pred = F.interpolate(pred, size=gt.shape, mode='bilinear', align_corners=True)
            result = pred.sigmoid().cpu().numpy().squeeze()
            result = (result - result.min() + 1e-20) / (result.max() - result.min() + 1e-20)

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

            tbar.set_description('mIOU: {:.4f}, Average MAE: {:.4f}, Mean DICE: {:.4f}, '
                                 'S-Measure {:.4f}, MAX E-Measure {:.4f}'
                                 .format(np.mean(Miou), np.mean(Amae), np.mean(Mdice), np.mean(Smeasure),
                                         np.mean(Emeasure)))

'''
Dataset   |  mIOU  | Average MAE | Mean DICE | S-Measure | Max E-Measure |
Kvasi-SEG | 0.8536  |   0.0269     |   0.9100   |   0.9165   |     0.9527     |
CVC-612   | 0.8363  |   0.0149     |   0.8905   |   0.9223   |     0.9551     |
ETIS      | 0.6036  |   0.0166     |   0.6840   |   0.8133   |     0.8576     |
Endoscene | 0.8273  |   0.0059     |   0.8995   |   0.9335   |     0.9734     |
CVC-Colon | 0.6967  |   0.0295     |   0.7831   |   0.8541   |     0.9007    |
'''
