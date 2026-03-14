# Source: https://github.com/pytorch/vision/blob/master/references/segmentation/transforms.py
import numpy as np
from PIL import Image
import random

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class Compose3(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target, target2):
        for t in self.transforms:
            image, target, target2 = t(image, target, target2)
        return image, target, target2


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class Normalize(T.Normalize):
    def __call__(self, image, target):
        return super().__call__(image), target


class Normalize3(T.Normalize):
    def __call__(self, image, target, target2):
        return super().__call__(image), target, target2


class ColorJitter(T.ColorJitter):
    def __call__(self, image, target):
        return super().__call__(image), target

class ColorJitter3(T.ColorJitter):
    def __call__(self, image, target, target2):
        return super().__call__(image), target, target2


def label_to_tensor(lbl):
    """
    Reads a PIL pallet Image img and convert the indices to a pytorch tensor
    """
    return torch.as_tensor(np.array(lbl, np.int64, copy=False)) 


def label_to_pil_image(lbl):
    """
    Creates a PIL pallet Image from a pytorch tensor of labels
    """
    if not(isinstance(lbl, torch.Tensor) or isinstance(lbl, np.ndarray)):
        raise TypeError('lbl should be Tensor or ndarray. Got {}.'.format(type(lbl)))
    elif isinstance(lbl, torch.Tensor):
        if lbl.ndimension() != 2:
            raise ValueError('lbl should be 2 dimensional. Got {} dimensions.'.format(lbl.ndimension()))
        lbl = lbl.numpy()
    elif isinstance(lbl, np.ndarray):
        if lbl.ndim != 2:
            raise ValueError('lbl should be 2 dimensional. Got {} dimensions.'.format(lbl.ndim))

    im = Image.fromarray(lbl.astype(np.uint8), mode='P')
    im.putpalette([0xee, 0xee, 0xec, 0xfc, 0xaf, 0x3e, 0x2e, 0x34, 0x36, 0x20, 0x4a, 0x87, 0xa4, 0x0, 0x0] + [0] * 753)
    return im


class ToTensor(object):
    def __call__(self, image, label):
        return F.to_tensor(image), label_to_tensor(label)