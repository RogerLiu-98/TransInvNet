import argparse
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as D
from tqdm import tqdm

from TransInvNet.model.model import TransInvNet, CONFIGS
from TransInvNet.utils.dataloader import PolypDataset
from TransInvNet.utils.utils import clip_gradient, adjust_lr
from eval import Metrics


def set_seeds(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def loss_fn(pred, mask):
    mask = mask.type_as(pred)
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def train(opt, train_loader, model, optimizer, epoch, loss_fn):
    model.train()
    optimizer.zero_grad()
    loss_record = []
    tbar = tqdm(train_loader, desc='\r')
    for i, pack in enumerate(tbar, start=1):
        # ---- data preparation ----
        images, gts = pack
        images = images.cuda()
        gts = gts.float().cuda()
        # ---- forward ----
        pred = model(images)
        # ---- compute loss ----
        loss = loss_fn(pred, gts)
        # ---- record loss ----
        loss_record.append(float(loss.data))
        # ---- gradient accumulation ----
        loss /= opt.accumulation_step
        # ---- backward ----
        loss.backward()
        clip_gradient(optimizer, opt.clip)
        if i % opt.accumulation_step == 0:
            optimizer.step()
            optimizer.zero_grad()
        # ---- train visualization ----
        if i == 1 or i % 20 == 0 or i == len(train_loader):
            tbar.set_description('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Train loss {:.04f}'.
                                 format(datetime.now(), epoch, opt.epoch, i, len(train_loader),
                                        np.mean(loss_record)))
    save_path = 'outputs/{}/train/'.format(opt.output_path)
    os.makedirs(save_path, exist_ok=True)
    if epoch % 10 == 0:
        torch.save(model.state_dict(), save_path + 'TransInvNet-%d.pth' % epoch)
        print('[Saving model weight:]', save_path + 'TransInvNet-%d.pth' % epoch)


def test(opt, test_loader, model, epoch, loss_fn, best_loss, best_metrics):
    model.eval()
    loss_record = []
    mDice, mMae, mIou = [], [], []
    with torch.no_grad():
        tbar = tqdm(test_loader, desc='\r')
        for i, pack in enumerate(tbar, start=1):
            # ---- data preparation ----
            images, gts = pack
            images = images.cuda()
            gts = gts.cuda()
            # ---- forward ----
            pred = model(images)
            # ---- compute loss ----
            loss = loss_fn(pred, gts)
            # --- compute metric ----
            pred = pred.sigmoid().cpu().numpy().squeeze()
            pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
            mask = gts.cpu().numpy().squeeze()
            metrics = Metrics(pred, mask)
            iou, mae, dice = metrics.calculate_iou(), metrics.calculate_mae(), metrics.calculate_dice()
            # ---- record loss and metrics ----
            loss_record.append(float(loss.data))
            mDice.append(dice)
            mMae.append(mae)
            mIou.append(iou)
            tbar.set_description('{} Epoch [{:03d}/{:03d}], Step[{:03d}/{:03d}], '
                                 'Test loss {:.04f}, Test mean dice {:.04f}, '
                                 'Test mean abs error {:.04f}, Test mean iou {:.04f}'.
                                 format(datetime.now(), epoch, opt.epoch, i, len(test_loader),
                                        np.mean(loss_record), np.mean(mDice),
                                        np.mean(mMae), np.mean(mIou)))
    current_loss = np.mean(loss_record)
    metrics = (np.mean(mDice) + np.mean(mIou) + (1 - np.mean(mMae))) / 3
    if current_loss < best_loss[-1] or metrics > best_metrics[-1]:
        best_loss.append(current_loss)
        best_metrics.append(metrics)
        save_path = 'outputs/{}/train/'.format(opt.output_path)
        torch.save(model.state_dict(), save_path + 'TransInvNet-best.pth')
        print('[Saving model weight], current best test loss is {:.04f}, current best mean dice is {:.04f}, '
              'current best mean abs error is {:.04f}, current best mean iou is {:.04f}'
              .format(current_loss, np.mean(mDice), np.mean(mMae), np.mean(mIou)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int,
                        default=100, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int,
                        default=4, help='training batch size')
    parser.add_argument('--accumulation_step', type=int,
                        default=4, help='accumulation step to implement gradient accumulation')
    parser.add_argument('--img_size', type=int,
                        default=352, help='training dataset size')
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    parser.add_argument('--cfg', type=str,
                        default='ViT-B_16', help='configs for model, choose from ViT-B_8, '
                                                 'ViT-B_16, ViT-B_32, ViT-L_16, ViT-L_32')
    parser.add_argument('--train_path', type=str,
                        default='datasets/polyp-dataset/TrainDataset', help='path to train dataset')
    parser.add_argument('--test_path', type=str,
                        default='datasets/polyp-dataset/TestDataset/Kvasir', help='path to test dataset')
    parser.add_argument('--output_path', type=str,
                        default='exp{}'.format(datetime.now().strftime('%m%d%H%M')), help='path to output')
    parser.add_argument('--seed', type=int,
                        default=1234, help='random seed')
    opt = parser.parse_args()

    set_seeds(seed=opt.seed)  # set seed

    # ---- prepare dataset ----
    train_dataset = PolypDataset(opt.train_path, image_dir='images', mask_dir='masks',
                                 new_size=(opt.img_size, opt.img_size))
    test_dataset = PolypDataset(opt.test_path, image_dir='images', mask_dir='masks',
                                new_size=(opt.img_size, opt.img_size))
    train_loader = D.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    test_loader = D.DataLoader(test_dataset, batch_size=1, shuffle=True)

    # ---- build models ----
    torch.cuda.set_device(0)  # set your gpu device
    cfg = CONFIGS[opt.cfg]
    model = TransInvNet(cfg, opt.img_size, vis=True, pretrained=True).cuda()

    # ---- flops and params ----
    # from TransInvNet.utils.utils import CalParams
    # x = torch.randn(1, 3, 352, 352).cuda()
    # CalParams(model, x)

    rednet_params = model.rednet.parameters()
    vit_params = model.transformer.parameters()
    decoder_params = model.decoder.parameters()

    optimizer = torch.optim.AdamW([
        {"params": rednet_params},
        {"params": vit_params},
        {"params": decoder_params, "lr": opt.lr * 10},
    ], opt.lr, weight_decay=5e-4)

    print("#" * 20, "Start Training", "#" * 20)

    best_loss, best_metrics = [np.inf], [-np.inf]
    for epoch in range(1, opt.epoch + 1):
        adjust_lr(optimizer, epoch, opt.epoch, 0.9)
        train(opt, train_loader, model, optimizer, epoch, loss_fn)
        test(opt, test_loader, model, epoch, loss_fn, best_loss, best_metrics)