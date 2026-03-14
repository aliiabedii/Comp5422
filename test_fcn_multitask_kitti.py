import torch
import numpy as np

from .models import FCN_MT 
from .utils import load_kitti_data
from . import dense_transforms
import torch.utils.tensorboard as tb


def test(args):
    from os import path
    """
    Your code here
    Hint: load the saved checkpoint of your model, and perform evaluation for both segmentation and depth estimation tasks 
          on the provided images of the KITTI dataset
    """


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    test(args)
