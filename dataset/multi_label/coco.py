import sys

import torch.utils.data as data
import json
import os
import subprocess
from PIL import Image
import numpy as np
import torch
import pickle
import logging

# from util import *


urls = {'train_img': 'http://images.cocodataset.org/zips/train2014.zip',
        'val_img': 'http://images.cocodataset.org/zips/val2014.zip',
        'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip'}


def categoty_to_idx(category):
    cat2idx = {}
    for cat in category:
        cat2idx[cat] = len(cat2idx)
    return cat2idx


class COCO14(data.Dataset):

    def __init__(self, cfg, split, transform=None, target_transform=None):

        root_path = '/mnt/data1/jiajian/datasets/coco14'
        self.img_dir = os.path.join(root_path, f'{split}2014')
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        list_path = os.path.join(root_path, 'ml_anno', f'coco14_{self.split}_anno.pkl')
        anno = pickle.load(open(list_path, 'rb+'))
        self.img_id = anno['image_name']
        self.label = anno['labels']
        self.img_idx = range(len(self.img_id))

        self.cat2idx = json.load(open(os.path.join(root_path, 'data', 'category.json'), 'r'))

        self.attr_id = list(self.cat2idx.keys())
        self.attr_num = len(self.cat2idx)

        # just for aligning with pedestrian attribute dataset
        self.eval_attr_num = len(self.cat2idx)

    def __len__(self):
        return len(self.img_id)

    def __getitem__(self, index):

        imgname, gt_label, imgidx = self.img_id[index], self.label[index], self.img_idx[index]
        imgpath = os.path.join(self.img_dir, imgname)
        img = Image.open(imgpath).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        gt_label = gt_label.astype(np.float32)

        if self.target_transform:
            gt_label = gt_label[self.target_transform]

        return img, gt_label, imgname

