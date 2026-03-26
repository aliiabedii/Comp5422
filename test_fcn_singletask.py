import torch
import numpy as np

from .models import FCN_ST 
from .utils import load_dense_data, ConfusionMatrix
from . import dense_transforms
import torch.utils.tensorboard as tb


def test(args):
    from os import path
    """
    Your code here
    Hint: load the saved checkpoint of your single-task model, and perform evaluation for the segmentation task
    Hint: use the ConfusionMatrix for you to calculate accuracy, mIoU for the segmentation task
     
    """

    return accuracy, mIoU


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    test(args)
