import random
import torch
import numpy as np
import torchvision.transforms as T
from PIL import ImageDraw

from dataset.autoaug import AutoAugment

from randaugment import RandAugment


class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class CutoutPIL(object):
    def __init__(self, cutout_factor=0.5):
        self.cutout_factor = cutout_factor

    def __call__(self, x):
        img_draw = ImageDraw.Draw(x)
        h, w = x.size[0], x.size[1]  # HWC
        h_cutout = int(self.cutout_factor * h + 0.5)
        w_cutout = int(self.cutout_factor * w + 0.5)
        y_c = np.random.randint(h)
        x_c = np.random.randint(w)

        y1 = np.clip(y_c - h_cutout // 2, 0, h)
        y2 = np.clip(y_c + h_cutout // 2, 0, h)
        x1 = np.clip(x_c - w_cutout // 2, 0, w)
        x2 = np.clip(x_c + w_cutout // 2, 0, w)
        fill_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img_draw.rectangle([x1, y1, x2, y2], fill=fill_color)

        return x


def get_transform(cfg):
    height = cfg.DATASET.HEIGHT
    width = cfg.DATASET.WIDTH
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if cfg.DATASET.TYPE == 'multi_label':

        valid_transform = T.Compose([
            T.Resize([height, width]),
            T.ToTensor(),
            # normalize,
        ])

        if cfg.TRAIN.DATAAUG.TYPE == 'randaug':
            # train_transform = T.Compose([
            #     T.RandomApply([AutoAugment()], p=cfg.TRAIN.DATAAUG.AUTOAUG_PROB),
            #     T.Resize((height, width), interpolation=3),
            #     CutoutPIL(cutout_factor=0.5),
            #     T.RandomHorizontalFlip(),
            #     T.ToTensor(),
            # ])

            train_transform = T.Compose([
                T.Resize((height, width), interpolation=3),
                RandAugment(),
                CutoutPIL(cutout_factor=0.5),
                T.ToTensor(),
            ])
        elif cfg.TRAIN.DATAAUG.TYPE == 'normal':
            train_transform = T.Compose([
                T.Resize((height, width)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                # T.RandomErasing(scale=(0.1, 0.5))
                # normalize,
            ])
        else:
            assert False, f'current {cfg.TRAIN.DATAAUG.TYPE} augmentation has not been achieved'


    else:
        assert False, 'xxxxxxxx'

    return train_transform, valid_transform
