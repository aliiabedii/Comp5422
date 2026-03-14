import torch

import torch.nn as nn
import torch.nn.functional as F


class CNNClassifier(torch.nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        """
        Your code here
        Initialize CNNClassifier with ResNet50 backbone + MLP classifier
        """
        # raise NotImplementedError('CNNClassifier.__init__') 
        # 1. Load pretrained ResNet50 backbone
        resnet50 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        
        # 2. Remove the final fully connected layer and global avg pool
        # I'll use all layers except the final FC
        self.backbone = nn.Sequential(*list(resnet50.children())[:-2])
        # This gives us output shape: (B, 2048, H/32, W/32) due to strides in ResNet50
        
        # 3. Global Average Pooling (GAP): (B, 2048, H', W') → (B, 2048)
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # 4. MLP Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),      # 2048 → 1024
            nn.ReLU(inplace=True),       # ReLU activation
            nn.Dropout(0.5),             # Optional: helps prevent overfitting
            nn.Linear(1024, num_classes) # 1024 → 6 classes
        )

    def forward(self, x):
        """
        Your code here
        Forward pass
        @x: torch.Tensor of shape (B, 3, H, W)
        @return: logits of shape (B, num_classes)
        """
        
        # 1. Extract features using ResNet50 backbone
        features = self.backbone(x)  # (B, 2048, H/32, W/32)
        
         # 2. Global Average Pooling
        pooled = self.gap(features)  # (B, 2048, 1, 1)
        pooled = pooled.view(pooled.size(0), -1)  # (B, 2048)
        
        # 3. Pass through MLP classifier
        logits = self.classifier(pooled)  # (B, num_classes)
        
        return logits
    

class FCN_ST(torch.nn.Module):
    def __init__(self):
        super().__init__()
        """
        Your code here.
        Hint: The Single-Task FCN needs to output segmentation maps at a higher resolution
        Hint: Use up-convolutions
        Hint: Use skip connections
        Hint: Use residual connections
        Hint: Always pad by kernel_size / 2, use an odd kernel_size
        """
        raise NotImplementedError('FCN_ST.__init__')

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,H,W))
        @return: torch.Tensor((B,C,H,W)), C is the number of classes for segmentation.
        Hint: Input and output resolutions need to match, use output_padding in up-convolutions, crop the output
              if required (use z = z[:, :, :H, :W], where H and W are the height and width of a corresponding CNNClassifier
              convolution
        """
        raise NotImplementedError('FCN_ST.forward')


class FCN_MT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        """
        Your code here.
        Hint: The Multi-Task FCN needs to output both segmentation and depth maps at a higher resolution
        Hint: Use up-convolutions
        Hint: Use skip connections
        Hint: Use residual connections
        Hint: Always pad by kernel_size / 2, use an odd kernel_size
        """
        raise NotImplementedError('FCN_MT.__init__')

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,H,W))
        @return: torch.Tensor((B,C,H,W)), C is the number of classes for segmentation
        @return: torch.Tensor((B,1,H,W)), 1 is one channel for depth estimation
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        Hint: Input and output resolutions need to match, use output_padding in up-convolutions, crop the output
              if required (use z = z[:, :, :H, :W], where H and W are the height and width of a corresponding strided
              convolution
        """
        raise NotImplementedError('FCN_MT.forward')


class SoftmaxCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftmaxCrossEntropyLoss, self).__init__()
        self.weight = weight # optional class wights for imbalanced datasets
        self.size_average = size_average # whether to average the loss over the batch

    def forward(self, inputs, targets):

        """
        Your code here
        Hint: inputs (prediction scores), targets (ground-truth labels)
        Hint: Implement a Softmax-CrossEntropy loss for classification
        Hint: return loss, F.cross_entropy(inputs, targets)
        """
        
        """
        Compute Softmax Cross-Entropy Loss from scratch
        
        @inputs: torch.Tensor of shape (B, C) or (B, C, H, W) - raw logits
        @targets: torch.Tensor of shape (B,) or (B, H, W) - integer class labels
        
        @return: scalar loss (if size_average=True) or per-sample loss
        """
        
        # Handle both 2D (classification) and 4D (segmentation) inputs
        if inputs.dim() == 4:
            # Reshape for pixel-wise classification: (B, C, H, W) → (B*H*W, C)
            B, C, H, W = inputs.shape
            inputs = inputs.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
            targets = targets.reshape(-1)  # (B*H*W,)
        
        # Numerical stability: subtract max logit before exp
        # This prevents overflow in exp()
        max_logits = inputs.max(dim=1, keepdim=True)[0]  # (N, 1)
        stable_inputs = inputs - max_logits  # (N, C)
        
        # Compute log-softmax: log(exp(x_i) / sum_j(exp(x_j)))
        exp_inputs = torch.exp(stable_inputs)  # (N, C)
        sum_exp = exp_inputs.sum(dim=1, keepdim=True)  # (N, 1)
        log_softmax = stable_inputs - torch.log(sum_exp)  # (N, C)
        
        # Gather the log-probability for the correct class
        # targets should be integer class labels
        N = inputs.size(0)
        correct_log_probs = log_softmax[range(N), targets]  # (N,)
        
        # Negative log-likelihood: -log(p_correct)
        loss = -correct_log_probs  # (N,)
        
        # Apply class weights if provided
        if self.weight is not None:
            # weight should be a tensor of shape (C,)
            weights_for_targets = self.weight[targets]  # (N,)
            loss = loss * weights_for_targets
        
        # Average or sum over batch
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
        # raise NotImplementedError('SoftmaxCrossEntropyLoss.__init__')


model_factory = {
    'cnn': CNNClassifier,
    'fcn_st': FCN_ST,
    'fcn_mt': FCN_MT
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
