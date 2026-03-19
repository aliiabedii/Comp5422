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
        
        
        """
        Initialize the Dense Cityscapes dataset
        Args:
            dataset_path: root path to DenseCityscapesDataset
            
            transform: transforms to apply
        """
        
        
        
        self.dataset_path = dataset_path
        self.transform = transform
        
        
        # Constants for depth calculation (from assignment section 4.1)
        self.baseline = 0.222384 # B
        self.focal_length = 2273.82 # f
        
         # Paths to each modality
        self.image_dir = os.path.join(dataset_path, 'image')
        self.label_dir = os.path.join(dataset_path, 'label')
        self.depth_dir = os.path.join(dataset_path, 'depth')  # This contains disparity
        
         # Get all image files (they should have corresponding label and depth files)
        self.image_files = sorted(glob(os.path.join(self.image_dir, '*.npy')))
        self.label_files = sorted(glob(os.path.join(self.label_dir, '*.npy')))
        self.depth_files = sorted(glob(os.path.join(self.depth_dir, '*.npy')))
        
        # Verify we have the same number of files
        assert len(self.image_files) == len(self.label_files) == len(self.depth_files), \
            f"Number of files mismatch: images={len(self.image_files)}, labels={len(self.label_files)}, depths={len(self.depth_files)}"
        
        print(f"✅ Loaded {len(self.image_files)} samples from {dataset_path}")

        #raise NotImplementedError('DenseCityscapesDataset.__init__')

    def __len__(self):

        """
        Your code here
        """
        # Return total number of samples
        return len(self.image_files)
        #raise NotImplementedError('DenseCityscapesDataset.__len__')

    def __getitem__(self, idx):

        """
        Hint: generate samples for training
        Hint: return image, semantic_GT, and depth_GT
        """
        
        # Load numpy files
        image_np = np.load(self.image_files[idx])
        semantic_np = np.load(self.label_files[idx])
        disparity_np = np.load(self.depth_files[idx])
        
        # Handle invalid labels: convert -1 to 255 (ignore_index for CrossEntropyLoss)
        semantic_np = semantic_np.copy()  # Make a copy to avoid modifying original
        semantic_np[semantic_np == -1] = 255
        
        # Handle disparity conversion as per instructions
        # disparity_np shape might be (H, W, 1), take first channel if needed
        if len(disparity_np.shape) == 3:
            disparity = disparity_np[:, :, 0]
        else:
            disparity = disparity_np
            
        # Convert disparity as per CityScape formula
        # (float(p)-1.) / 256, and we have *65535 because of .npy format
        disparity = (disparity * 65535 - 1) / 256
        
        # Calculate depth: depth = (B * f) / disparity
        # Avoid division by zero (invalid disparity values)
        valid_mask = disparity > 0
        depth = np.zeros_like(disparity, dtype=np.float32)
        depth[valid_mask] = (self.baseline * self.focal_length) / disparity[valid_mask]
        
        # Handle image conversion
        # Image is likely in [0, 1] range, convert to uint8 for PIL
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        
        # Ensure image is in HWC format for PIL
        if image_np.shape[0] == 3:  # CHW format
            image_np = np.transpose(image_np, (1, 2, 0))
        
        # Convert to PIL Image
        image_pil = Image.fromarray(image_np)
        
        # Convert semantic mask to PIL Image for processing with transforms
        # Values: 0-18 are valid classes, 255 is ignore_index
        semantic_pil = Image.fromarray(semantic_np.astype(np.uint8), mode='L')
        
        # Convert depth to PIL Image for processing with transforms
        # Normalize depth to [0, 255] range for PIL
        depth_min, depth_max = depth.min(), depth.max()
        if depth_max > depth_min and not np.isnan(depth_max):
            depth_normalized = ((depth - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
        else:
            depth_normalized = depth.astype(np.uint8)
        depth_pil = Image.fromarray(depth_normalized, mode='L')
        
        # Apply transforms (which handle image, semantic, depth together as PIL Images)
        if self.transform:
            image_tensor, semantic_tensor, depth_tensor = self.transform(image_pil, semantic_pil, depth_pil)
        else:
            # Fallback to basic transforms
            from torchvision import transforms
            to_tensor = transforms.ToTensor()
            image_tensor = to_tensor(image_pil)
            semantic_tensor = torch.from_numpy(semantic_np).long()
            depth_tensor = torch.from_numpy(depth).float()
        
        return image_tensor, semantic_tensor, depth_tensor

        #raise NotImplementedError('DenseCityscapesDataset.__getitem__')
    

class DenseKITTIDataset(Dataset):
    """
    Dataset for KITTI test samples (cross-dataset evaluation)
    Files are PNG images directly in the folder
    """
    def __init__(self, dataset_path, transform=dense_transforms.ToTensor3()):
        """
        Initialize KITTI dataset
        
        Args:
            dataset_path: path to folder containing KITTI test PNGs
            transform: transforms to apply
        """
        self.dataset_path = dataset_path
        self.transform = transform
        
        # Get all PNG files in the directory
        self.image_files = sorted(glob(os.path.join(dataset_path, '*.png')))
        
        if len(self.image_files) == 0:
            # Try other extensions if no PNGs found
            self.image_files = sorted(glob(os.path.join(dataset_path, '*.jpg')))
        
        if len(self.image_files) == 0:
            raise ValueError(f"No image files found in {dataset_path}")
        
        print(f"✅ Loaded {len(self.image_files)} KITTI test samples (PNG format)")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Load and return (image, dummy_semantic, dummy_depth)
        For KITTI cross-dataset evaluation, we only need images
        """
        # Load PNG image
        image_path = self.image_files[idx]
        image_pil = Image.open(image_path).convert('RGB')
        
        # Get image dimensions for dummy tensors
        W, H = image_pil.size
        
        # Create dummy semantic (all zeros) and depth (all zeros)
        dummy_semantic = Image.fromarray(np.zeros((H, W), dtype=np.uint8))
        dummy_depth = np.zeros((H, W), dtype=np.float32)
        
        # Apply transforms
        if self.transform:
            image_tensor, semantic_tensor, depth_tensor = self.transform(image_pil, dummy_semantic, dummy_depth)
        else:
            from torchvision import transforms
            to_tensor = transforms.ToTensor()
            image_tensor = to_tensor(image_pil)
            semantic_tensor = torch.from_numpy(np.array(dummy_semantic)).long()
            depth_tensor = torch.from_numpy(dummy_depth).float()
        
        return image_tensor, semantic_tensor, depth_tensor


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


def load_dense_data(dataset_path, num_workers=0, batch_size=32, shuffle=True, drop_last=True, transform=None, **kwargs):
    """
    Load dense data from a folder containing 'image', 'label', 'depth' subfolders
    """
    print(f"load_dense_data received transform: {transform}")
    print(f"transform type: {type(transform)}")
    
    if transform is None:
        transform = dense_transforms.ToTensor()
        print(f"Using default transform: {transform}")
    
    dataset = DenseCityscapesDataset(dataset_path, transform=transform, **kwargs)
    print(f"Dataset created with transform: {dataset.transform}")
    
    return DataLoader(dataset, 
                     num_workers=num_workers, 
                     batch_size=batch_size, 
                     shuffle=shuffle, 
                     drop_last=drop_last)


def load_kitti_data(dataset_path, num_workers=0, batch_size=1, shuffle=False, **kwargs):
    """
    Load KITTI data for cross-dataset evaluation
    
    Args:
        dataset_path: path to KITTI dataset folder
        num_workers: number of data loading workers
        batch_size: batch size (use 1 for visualization)
        shuffle: whether to shuffle the data
        **kwargs: additional arguments passed to DenseKITTIDataset (like transform)
    """
    dataset = DenseKITTIDataset(dataset_path, **kwargs)
    return DataLoader(dataset, 
                     num_workers=num_workers, 
                     batch_size=batch_size, 
                     shuffle=shuffle)


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


def load_class_weights(weight_file='classweight.cls'):
    """
    Load class weights for semantic segmentation from the provided file
    
    The file format is expected to have one weight per line, optionally with labels:
    ID class: weight
    0 road: 3.29
    1 sidewalk: 21.9
    ...
    
    Args:
        weight_file: path to the classweight.cls file
    
    Returns:
        torch.Tensor of shape (19,) containing class weights
    """
    weights = []
    try:
        with open(weight_file, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and header
                if not line or line.startswith('ID'):
                    continue
                
                # Extract the weight (last part after colon or last word)
                if ':' in line:
                    weight = float(line.split(':')[-1].strip())
                else:
                    # Try to parse the last token as float
                    parts = line.split()
                    if parts:
                        weight = float(parts[-1])
                    else:
                        continue
                weights.append(weight)
        
        # Verify we have 19 weights
        if len(weights) != 19:
            print(f"⚠️  Warning: Expected 19 weights, got {len(weights)}")
            # Pad or truncate if necessary? Better to raise error
            if len(weights) < 19:
                raise ValueError(f"Only {len(weights)} weights found in {weight_file}")
            else:
                weights = weights[:19]
        
        print(f"✅ Loaded {len(weights)} class weights from {weight_file}")
        return torch.tensor(weights, dtype=torch.float32)
        
    except FileNotFoundError:
        print(f"❌ Weight file {weight_file} not found!")
        raise
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        raise