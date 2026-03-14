import torch
import numpy as np

from .models import FCN_MT 
from .utils import load_dense_data, ConfusionMatrix, DepthError
from . import dense_transforms
import torch.utils.tensorboard as tb


def test(args):
    from os import path
    """
    Your code here
    Hint: load the saved checkpoint of your model, and perform evaluation for both segmentation and depth estimation tasks
    Hint: use the ConfusionMatrix for you to calculate accuracy, mIoU for the segmentation task
    Hint: use DepthError for you to calculate rel, a1, a2, and a3 for the depth estimation task. 
    """

    return accuracy, mIoU, rel, a1, a2, a3


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    test(args)
