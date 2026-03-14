import os
import torch

import numpy as np

from PIL import Image
from glob import glob
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F

from . import dense_transforms


class VehicleClassificationDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        """
        Initializing the dataset by loading the image paths and their corresponding labels.
        Hint: load your data from provided dataset (VehicleClassificationDataset) to train your designed model
        """
        # e.g., Bicycle 0, Car 1, Taxi 2, Bus 3, Truck 4, Van 5
        self.data = []
        self.labels = []
        self.transform = transform
        class_map = {
            'Bicycle': 0,
            'Car': 1,
            'Taxi': 2,
            'Bus': 3,
            'Truck': 4,
            'Van': 5
        }
        # Iterate through each class folder
        for class_name, class_label in class_map.items():
            class_folder = os.path.join(dataset_path, class_name)
            if os.path.exists(class_folder):
                
                # Get all image files in the folder
                image_paths = glob(os.path.join(class_folder, '*.jpg'))
                image_paths += glob(os.path.join(class_folder, '*.JPG'))
                
                # Add each image path with its label
                for img_path in image_paths:
                    self.data.append(img_path)
                    self.labels.append(class_label)
                    
        if len(self.data) == 0:
            raise ValueError(f"No images found in the dataset path: {dataset_path}")
        
        print(f"✅ Loaded {len(self.data)} images from {dataset_path}")
        
        # CRITICAL: Resize + Normalize for ResNet50
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),  # Resize all images to 224x224
                transforms.ToTensor(),           # Convert to tensor [0, 1]
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])                
            ])
        """
        when I don't normalize or resize, DataLoader cannot load the images
        later I should add Data augmentation here, but for now I just want to make sure the data loading works.
        """
    

        
         

    def __len__(self):
        """Return the total number of images in the dataset"""
        return len(self.data)

        # raise NotImplementedError('VehicleClassificationDataset.__len__')

    def __getitem__(self, idx):
        """
        Your code here
        Hint: generate samples for training
        Hint: return image, and its image-level class label
        """
        # Get image path and label
        img_path = self.data[idx]
        label = self.labels[idx]
        # Load and transform the image using PIL        
        
        image = Image.open(img_path).convert('RGB')
        
        # apply transform (resize + normalize)          
        if self.transform:
            image = self.transform(image)
            
        return image, label

class DenseCityscapesDataset(Dataset):
    """
    HINT:
    Before translating the disparity into real depth, 
    we need to load the disparity from .npy files correctly. 
    
    Example:
        value = np.load('0.npy') [:,:,0]
        disparity = (value * 65535 - 1) / 256
        depth = (Baseline * focal_length) / disparity 
    
    According to the readme of the CityScape dataset 
    (https://github.com/mcordts/cityscapesScripts/blob/master/README.md#dataset-structure),
    we need to load disparity with (float(p)-1.) / 256. The *65535 operation is 
    because we provide data in the .npy format, not the original 16-bit png. 
    Please also note that there are some invalid depth values (not positive)
    in the ground truths caused by sensors. They should be masked out in the 
    training and evaluation. 

    """
    def __init__(self, dataset_path, transform=dense_transforms.ToTensor()):
        """
        Your code here
        """
        """
        Initialize thedence Cityscapes dataset by loading the image paths
        """
        self.dataset_path = dataset_path
        self.transform = transform
        
        # Constants for depth calculation (from assignment section 4.1)
        self.baseline = 0.222384 # B
        self.focal_length = 2273.82 # f
        
        # Load all sample file paths
        self.samples = []
        raise NotImplementedError('DenseCityscapesDataset.__init__')

    def __len__(self):

        """
        Your code here
        """
        raise NotImplementedError('DenseCityscapesDataset.__len__')

    def __getitem__(self, idx):

        """
        Hint: generate samples for training
        Hint: return image, semantic_GT, and depth_GT
        """
        raise NotImplementedError('DenseCityscapesDataset.__getitem__')
    

class DenseKITTIDataset(Dataset):
    def __init__(self, dataset_path, transform=dense_transforms.ToTensor()):
        """
        Your code here
        """
        raise NotImplementedError('DenseKITTIDataset.__init__')

    def __len__(self):
        """
        Your code here
        """
        raise NotImplementedError('DenseKITTIDataset.__len__')

    def __getitem__(self, idx):
        """
        Your code here
        """
        raise NotImplementedError('DenseKITTIDataset.__getitem__')


class DenseVisualization():
    def __init__(self, img, depth, segmentation):
        self.img = img
        self.depth = depth
        self.segmentation = segmentation

    def __visualizeitem__(self):
        """
        Your code here
        Hint: you can visualize your model predictions and save them into images. 
        """
        raise NotImplementedError('DenseVisualization.__visualizeitem__')


def load_data(dataset_path, num_workers=0, batch_size=128, **kwargs):
    dataset = VehicleClassificationDataset(dataset_path, **kwargs)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


def load_dense_data(dataset_path, num_workers=0, batch_size=32, **kwargs):
    dataset = DenseCityscapesDataset(dataset_path, **kwargs)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


def load_kitti_data(dataset_path, num_workers=0, batch_size=32, **kwargs):
    dataset = DenseKITTIDataset(dataset_path, **kwargs)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


def _one_hot(x, n):
    return (x.view(-1, 1) == torch.arange(n, dtype=x.dtype, device=x.device)).int()


class ConfusionMatrix(object):
    def _make(self, preds, labels):
        label_range = torch.arange(self.size, device=preds.device)[None, :]
        preds_one_hot, labels_one_hot = _one_hot(preds, self.size), _one_hot(labels, self.size)
        return (labels_one_hot[:, :, None] * preds_one_hot[:, None, :]).sum(dim=0).detach()

    def __init__(self, size=5):
        """
        This class builds and updates a confusion matrix.
        :param size: the number of classes to consider
        """
        self.matrix = torch.zeros(size, size)
        self.size = size

    def add(self, preds, labels):
        """
        Updates the confusion matrix using predictions `preds` (e.g. logit.argmax(1)) and ground truth `labels`
        """
        self.matrix = self.matrix.to(preds.device)
        self.matrix += self._make(preds, labels).float()

    @property
    def class_iou(self):
        true_pos = self.matrix.diagonal()
        return true_pos / (self.matrix.sum(0) + self.matrix.sum(1) - true_pos + 1e-5)

    @property
    def iou(self):
        return self.class_iou.mean()

    @property
    def global_accuracy(self):
        true_pos = self.matrix.diagonal()
        return true_pos.sum() / (self.matrix.sum() + 1e-5)

    @property
    def class_accuracy(self):
        true_pos = self.matrix.diagonal()
        return true_pos / (self.matrix.sum(1) + 1e-5)

    @property
    def average_accuracy(self):
        return self.class_accuracy.mean()

    @property
    def per_class(self):
        return self.matrix / (self.matrix.sum(1, keepdims=True) + 1e-5)


class DepthError(object):
    def __init__(self, gt, pred):
        """
        This class builds and updates a confusion matrix.
        :param size: the number of classes to consider
        """
        self.gt = gt
        self.pred = pred

    @property
    def compute_errors(self):
        """Computation of error metrics between predicted and ground truth depths
        """
        thresh = np.maximum((self.gt / self.pred), (self.pred / self.gt))
        a1 = (thresh < 1.25     ).mean()
        a2 = (thresh < 1.25 ** 2).mean()
        a3 = (thresh < 1.25 ** 3).mean()

        # rmse = (self.gt - self.pred) ** 2
        # rmse = np.sqrt(rmse.mean())

        # rmse_log = (np.log(self.gt) - np.log(self.pred)) ** 2
        # rmse_log = np.sqrt(rmse_log.mean())

        abs_rel = np.mean(np.abs(self.gt - self.pred) / self.gt)

        # sq_rel = np.mean(((self.gt - self.pred) ** 2) / self.gt)

        return abs_rel, a1, a2, a3
