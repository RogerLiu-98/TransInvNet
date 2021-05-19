import pathlib

import albumentations as A
import numpy as np
import torch.utils.data as data
from PIL import Image
from albumentations.pytorch.transforms import ToTensorV2
from tqdm import tqdm


class PolypDataset(data.Dataset):

    def __init__(self, dataset_dir='', image_dir='', mask_dir='', new_size=(352, 352)):
        super(PolypDataset, self).__init__()
        self.dataset_dir = pathlib.Path(dataset_dir)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.new_size = new_size
        self.images = []
        self.masks = []
        self.transform = A.Compose([
            A.Normalize(mean=[0.497, 0.302, 0.216],
                        std=[0.320, 0.217, 0.173]),
            ToTensorV2(),
        ])
        self.load_images()

    def load_images(self):
        image_files = [i for i in (self.dataset_dir / self.image_dir).iterdir()]
        mask_files = [i for i in (self.dataset_dir / self.mask_dir).iterdir()]
        assert len(image_files) == len(mask_files), 'The number of images does not match the number masks!'

        tbar = tqdm(zip(image_files, mask_files), total=len(image_files), desc='\r')
        for i, pack in enumerate(tbar, start=1):
            image_name, mask_name = pack
            # Load image
            image = Image.open(image_name).convert('RGB').resize(self.new_size)
            image = np.asarray(image)

            # Load mask
            mask = Image.open(mask_name).convert('1').resize(self.new_size)
            mask = np.asarray(mask).astype(np.uint8)

            self.images.append(image)
            self.masks.append(mask)

            tbar.set_description('Processing [{}/{}] image'.format(i, len(image_files)))

    def __getitem__(self, index):
        image, mask = self.images[index], self.masks[index]
        augments = self.transform(image=image, mask=mask)
        return augments['image'], augments['mask'][None]

    def __len__(self):
        return len(self.images)
