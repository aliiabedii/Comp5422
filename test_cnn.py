from .models import CNNClassifier
from .utils import ConfusionMatrix, load_data
import torch
import torchvision
import torch.utils.tensorboard as tb


def test(args):
    from os import path
    model = CNNClassifier()

    """
    Your code here
    Hint: load the saved checkpoint of your model, and perform evaluation for the vehicle classification task
    Hint: use the ConfusionMatrix for you to calculate accuracy
    """
    return accuracy


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    test(args)
