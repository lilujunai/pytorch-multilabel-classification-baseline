import torch
import numpy as np
import torchvision.transforms as T

from dataset.autoaug import AutoAugment


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

        if cfg.TRAIN.DATAAUG.TYPE == 'autoaug':
            train_transform = T.Compose([
                T.RandomApply([AutoAugment()], p=cfg.TRAIN.DATAAUG.AUTOAUG_PROB),
                T.Resize((height, width), interpolation=3),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ])
        else:
            train_transform = T.Compose([
                T.Resize((height, width)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                # T.RandomErasing(scale=(0.1, 0.5))
                # normalize,
            ])
    else:
        assert False, 'xxxxxxxx'

    return train_transform, valid_transform
