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
    def __init__(self, num_classes=19):
        super().__init__()
        """
        Your code here.
        Hint: The Single-Task FCN needs to output segmentation maps at a higher resolution
        Hint: Use up-convolutions
        Hint: Use skip connections
        Hint: Use residual connections
        Hint: Always pad by kernel_size / 2, use an odd kernel_size
        """
        # 1. Load pretrained ResNet50 backbone
        resnet50 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        
        # 2. Extract encoder layers
        self.encoder1 = nn.Sequential(
            resnet50.conv1,
            resnet50.bn1,
            resnet50.relu
        )  # Output: 64 channels, H/2, W/2
        
        self.encoder2 = nn.Sequential(
            resnet50.maxpool,
            resnet50.layer1
        )  # Output: 256 channels, H/4, W/4
        
        self.encoder3 = resnet50.layer2  # Output: 512 channels, H/8, W/8
        self.encoder4 = resnet50.layer3  # Output: 1024 channels, H/16, W/16
        self.encoder5 = resnet50.layer4  # Output: 2048 channels, H/32, W/32
        
        # 3. Decoder with up-convolutions and skip connections
        self.decoder5 = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(1024 + 1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(512 + 512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(256 + 256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(128 + 64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # 4. Final classifier
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        """
        Forward pass for single-task semantic segmentation
        """
        B, C, H, W = x.shape
        
        # Encoder
        e1 = self.encoder1(x)      # (B, 64, H/2, W/2)
        e2 = self.encoder2(e1)      # (B, 256, H/4, W/4)
        e3 = self.encoder3(e2)      # (B, 512, H/8, W/8)
        e4 = self.encoder4(e3)      # (B, 1024, H/16, W/16)
        e5 = self.encoder5(e4)      # (B, 2048, H/32, W/32)
        
        # Decoder with skip connections
        d5 = self.decoder5(e5)               # (B, 1024, H/16, W/16)
        d5 = torch.cat([d5, e4], dim=1)       # (B, 2048, H/16, W/16)
        
        d4 = self.decoder4(d5)                # (B, 512, H/8, W/8)
        d4 = torch.cat([d4, e3], dim=1)       # (B, 1024, H/8, W/8)
        
        d3 = self.decoder3(d4)                # (B, 256, H/4, W/4)
        d3 = torch.cat([d3, e2], dim=1)       # (B, 512, H/4, W/4)
        
        d2 = self.decoder2(d3)                # (B, 128, H/2, W/2)
        d2 = torch.cat([d2, e1], dim=1)       # (B, 192, H/2, W/2)
        
        d1 = self.decoder1(d2)                # (B, 64, H, W)
        
        # Final classification
        out = self.classifier(d1)              # (B, num_classes, H, W)
        
        # Crop if needed
        if out.shape[2] != H or out.shape[3] != W:
            out = out[:, :, :H, :W]
        
        return out


class FCN_MT(torch.nn.Module):
    def __init__(self, num_classes=19):
        super().__init__()
        """
        Multi-Task FCN for both semantic segmentation and depth estimation
        Shares encoder backbone, but has two separate decoder heads:
        1. Segmentation head (same as FCN_ST)
        2. Depth prediction head (outputs 1 channel depth map)
        """
        
        # 1. Load pretrained ResNet50 backbone (shared encoder)
        resnet50 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        
        # === SHARED ENCODER (same as FCN_ST) ===
        # Encoder 1: initial conv + bn + relu (stride 2)
        self.encoder1 = nn.Sequential(
            resnet50.conv1,
            resnet50.bn1,
            resnet50.relu
        )  # Output: 64 channels, H/2, W/2
        
        # Encoder 2: maxpool + layer1 (stride 2 total)
        self.encoder2 = nn.Sequential(
            resnet50.maxpool,
            resnet50.layer1
        )  # Output: 256 channels, H/4, W/4
        
        # Encoder 3: layer2 (stride 2)
        self.encoder3 = resnet50.layer2  # Output: 512 channels, H/8, W/8
        
        # Encoder 4: layer3 (stride 2)
        self.encoder4 = resnet50.layer3  # Output: 1024 channels, H/16, W/16
        
        # Encoder 5: layer4 (stride 2)
        self.encoder5 = resnet50.layer4  # Output: 2048 channels, H/32, W/32
        
        # === SEGMENTATION HEAD (same as FCN_ST) ===
        # Decoder 5: upsample from 2048 → 1024
        self.seg_decoder5 = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        
        # Decoder 4: combine with encoder4 (1024) → 512
        self.seg_decoder4 = nn.Sequential(
            nn.ConvTranspose2d(1024 + 1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Decoder 3: combine with encoder3 (512) → 256
        self.seg_decoder3 = nn.Sequential(
            nn.ConvTranspose2d(512 + 512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Decoder 2: combine with encoder2 (256) → 128
        self.seg_decoder2 = nn.Sequential(
            nn.ConvTranspose2d(256 + 256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Decoder 1: combine with encoder1 (64) → 64
        self.seg_decoder1 = nn.Sequential(
            nn.ConvTranspose2d(128 + 64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Final segmentation classifier (19 classes)
        self.seg_classifier = nn.Conv2d(64, num_classes, kernel_size=1)
        
        # === DEPTH PREDICTION HEAD (separate decoder) ===
        # Depth decoder also starts from encoder5 (same shared features)
        self.depth_decoder5 = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        
        self.depth_decoder4 = nn.Sequential(
            nn.ConvTranspose2d(1024 + 1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        self.depth_decoder3 = nn.Sequential(
            nn.ConvTranspose2d(512 + 512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.depth_decoder2 = nn.Sequential(
            nn.ConvTranspose2d(256 + 256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.depth_decoder1 = nn.Sequential(
            nn.ConvTranspose2d(128 + 64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Final depth prediction (1 channel for depth values)
        # Using Tanh to constrain output range (helps with training stability)
        self.depth_head = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Tanh()  # Outputs between -1 and 1, will scale later
        )
        
        # Depth scaling factor (to map Tanh output to real depth range)
        self.depth_scale = 50.0  # Max depth ~50 meters
        
        # Store input size for cropping
        self.input_size = None

    def forward(self, x):
        B, C, H, W = x.shape
        
        # === SHARED ENCODER ===
        e1 = self.encoder1(x)      # (B, 64, H/2, W/2)
        e2 = self.encoder2(e1)      # (B, 256, H/4, W/4)
        e3 = self.encoder3(e2)      # (B, 512, H/8, W/8)
        e4 = self.encoder4(e3)      # (B, 1024, H/16, W/16)
        e5 = self.encoder5(e4)      # (B, 2048, H/32, W/32)
        
        # Store encoder output shapes for debugging
        encoder_shapes = {
            'e1': e1.shape,
            'e2': e2.shape,
            'e3': e3.shape,
            'e4': e4.shape,
            'e5': e5.shape
        }
        
        # === SEGMENTATION HEAD (with flexible size matching) ===
        # Decoder 5: upsample e5
        s5 = self.seg_decoder5(e5)  # Target: (B, 1024, H/16, W/16)
        
        # Ensure e4 matches s5 spatial dimensions
        if s5.shape[2:] != e4.shape[2:]:
            s5 = F.interpolate(s5, size=e4.shape[2:], mode='bilinear', align_corners=False)
        s5 = torch.cat([s5, e4], dim=1)  # (B, 2048, H/16, W/16)
        
        # Decoder 4
        s4 = self.seg_decoder4(s5)  # Target: (B, 512, H/8, W/8)
        
        # Ensure e3 matches s4 spatial dimensions
        if s4.shape[2:] != e3.shape[2:]:
            s4 = F.interpolate(s4, size=e3.shape[2:], mode='bilinear', align_corners=False)
        s4 = torch.cat([s4, e3], dim=1)  # (B, 1024, H/8, W/8)
        
        # Decoder 3
        s3 = self.seg_decoder3(s4)  # Target: (B, 256, H/4, W/4)
        
        # Ensure e2 matches s3 spatial dimensions
        if s3.shape[2:] != e2.shape[2:]:
            s3 = F.interpolate(s3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        s3 = torch.cat([s3, e2], dim=1)  # (B, 512, H/4, W/4)
        
        # Decoder 2
        s2 = self.seg_decoder2(s3)  # Target: (B, 128, H/2, W/2)
        
        # Ensure e1 matches s2 spatial dimensions
        if s2.shape[2:] != e1.shape[2:]:
            s2 = F.interpolate(s2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        s2 = torch.cat([s2, e1], dim=1)  # (B, 192, H/2, W/2)
        
        # Decoder 1
        s1 = self.seg_decoder1(s2)  # (B, 64, H, W)
        
        # Final classifier
        seg_out = self.seg_classifier(s1)  # (B, 19, H, W)
        
        # === DEPTH HEAD (with same flexible approach) ===
        # Decoder 5
        d5 = self.depth_decoder5(e5)
        if d5.shape[2:] != e4.shape[2:]:
            d5 = F.interpolate(d5, size=e4.shape[2:], mode='bilinear', align_corners=False)
        d5 = torch.cat([d5, e4], dim=1)
        
        # Decoder 4
        d4 = self.depth_decoder4(d5)
        if d4.shape[2:] != e3.shape[2:]:
            d4 = F.interpolate(d4, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d4 = torch.cat([d4, e3], dim=1)
        
        # Decoder 3
        d3 = self.depth_decoder3(d4)
        if d3.shape[2:] != e2.shape[2:]:
            d3 = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, e2], dim=1)
        
        # Decoder 2
        d2 = self.depth_decoder2(d3)
        if d2.shape[2:] != e1.shape[2:]:
            d2 = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, e1], dim=1)
        
        # Decoder 1
        d1 = self.depth_decoder1(d2)  # (B, 64, H, W)
        
        # Final depth prediction
        depth_out = self.depth_head(d1)
        depth_out = (depth_out + 1) / 2 * self.depth_scale  # Scale to meters
        
        # Final crop to ensure exact input size
        if seg_out.shape[2] != H or seg_out.shape[3] != W:
            seg_out = F.interpolate(seg_out, size=(H, W), mode='bilinear', align_corners=False)
            depth_out = F.interpolate(depth_out, size=(H, W), mode='bilinear', align_corners=False)
        
        return seg_out, depth_out
        
        
        
        
        
        


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
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.pth' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.pth' % model), map_location='cpu'))
    return r
