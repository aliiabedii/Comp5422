import torch
import numpy as np

from .models import FCN_ST, save_model
from .utils import load_dense_data, ConfusionMatrix, load_class_weights
from . import dense_transforms
import torch.utils.tensorboard as tb
import torch.optim as optim
import os


def train(args):
    from os import path
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = FCN_ST(num_classes=19).to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Setup loggers
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)
        print(f"Logging to: {args.log_dir}")

    """
    Your code here
    Hint: Use ConfusionMatrix, ConfusionMatrix.add(logit.argmax(1), label), ConfusionMatrix.iou to compute
          the overall IoU, where label are the batch labels, and logit are the logits of your classifier.
    Hint: Use dense_transforms for data augmentation. If you found a good data augmentation parameters for the CNN, use them here too.
    Hint: Use the log function below to debug and visualize your model
    """
    
    # ============================================
    # 1. Setup Data Loaders with Augmentation
    # ============================================
    
    # Training transforms with augmentation (make parameters accessible)
    # Training transforms with augmentation
    train_transform = dense_transforms.Compose3([
    dense_transforms.RandomHorizontalFlip3(flip_prob=args.flip_prob),  # ← CHANGE HERE
    dense_transforms.ColorJitter3(
        brightness=args.color_jitter[0], 
        contrast=args.color_jitter[1], 
        saturation=args.color_jitter[2], 
        hue=args.color_jitter[3]
        ),
        dense_transforms.ToTensor3(),
    ])
    
    # Validation transforms (no augmentation)
    val_transform = dense_transforms.ToTensor3()
    
    # Load datasets
    train_loader = load_dense_data(
        args.train_dir,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        transform=train_transform
    )
    
    val_loader = load_dense_data(
        args.val_dir,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        transform=val_transform
    )
    
    print(f"Train loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")
    
    # ============================================
    # 2. Setup Loss, Optimizer, and Class Weights
    # ============================================
    
    # Load class weights for imbalanced classes
    class_weights = load_class_weights(args.weight_file).to(device)
    print(f"Class weights loaded: {class_weights.shape}")
    
    # Use PyTorch's CrossEntropyLoss with weights and ignore_index=255
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
    
    # Flexible optimizer selection
    if args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    
    # Learning rate scheduler
    if args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.gamma, patience=args.patience)
    else:
        scheduler = None
    
    # ============================================
    # 3. Training Loop
    # ============================================
    
    num_epochs = args.num_epochs
    global_step = 0
    best_val_iou = 0.0
    
    for epoch in range(num_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*50}")
        
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        train_confusion = ConfusionMatrix(size=19)
        
        for batch_idx, (images, labels, _) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping if enabled
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            train_confusion.add(logits.argmax(1), labels)
            
            # Log training loss
            if train_logger is not None and global_step % args.log_freq == 0:
                train_logger.add_scalar('loss', loss.item(), global_step)
                
                # Log images every N steps
                if global_step % args.vis_freq == 0:
                    log(train_logger, images, labels, logits, global_step)
            
            global_step += 1
            
            if batch_idx % args.print_freq == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
        # Calculate training metrics
        train_loss = train_loss / len(train_loader)
        train_iou = train_confusion.iou.item()
        
        print(f"\nTrain Loss: {train_loss:.4f} | Train IoU: {train_iou:.4f}")
        
        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_confusion = ConfusionMatrix(size=19)
        
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                logits = model(images)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                val_confusion.add(logits.argmax(1), labels)
        
        # Calculate validation metrics
        val_loss = val_loss / len(val_loader)
        val_iou = val_confusion.iou.item()
        val_acc = val_confusion.global_accuracy.item()
        
        print(f"Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f} | Val Acc: {val_acc:.4f}")
        
        # Log epoch metrics
        if train_logger is not None:
            train_logger.add_scalar('epoch_loss', train_loss, epoch)
            train_logger.add_scalar('epoch_iou', train_iou, epoch)
        
        if valid_logger is not None:
            valid_logger.add_scalar('epoch_loss', val_loss, epoch)
            valid_logger.add_scalar('epoch_iou', val_iou, epoch)
            valid_logger.add_scalar('epoch_accuracy', val_acc, epoch)
        
        # Learning rate scheduler step
        if scheduler is not None:
            if args.scheduler == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current LR: {current_lr:.6f}")
        
        # Save best model based on validation IoU
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            save_model(model)
            print(f"✅ Model saved! Best Val IoU: {best_val_iou:.4f}")
    
    print(f"\n{'='*50}")
    print(f"Training Complete!")
    print(f"Best Validation IoU: {best_val_iou:.4f}")
    print(f"{'='*50}")
    
    save_model(model)
    return best_val_iou


def log(logger, imgs, lbls, logits, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    lbls: semantic label tensor
    logits: predicted logits tensor
    global_step: iteration
    """
    logger.add_image('image', imgs[0], global_step)
    logger.add_image('label', np.array(dense_transforms.label_to_pil_image(lbls[0].cpu()).
                                             convert('RGB')), global_step, dataformats='HWC')
    logger.add_image('prediction', np.array(dense_transforms.
                                                  label_to_pil_image(logits[0].argmax(dim=0).cpu()).
                                                  convert('RGB')), global_step, dataformats='HWC')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    
    # Required arguments
    parser.add_argument('--train_dir', type=str, required=True,
                        help='Path to training data (e.g., DenseCityscapesDataset/train)')
    parser.add_argument('--val_dir', type=str, required=True,
                        help='Path to validation data (e.g., DenseCityscapesDataset/val)')
    
    # Logging
    parser.add_argument('--log_dir', type=str, default='logs/fcn_st',
                        help='Directory for TensorBoard logs')
    parser.add_argument('--log_freq', type=int, default=10,
                        help='Logging frequency (steps)')
    parser.add_argument('--vis_freq', type=int, default=500,
                        help='Visualization frequency (steps)')
    parser.add_argument('--print_freq', type=int, default=10,
                        help='Print frequency (batches)')
    
    # Data
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--weight_file', type=str, default='classweight.cls',
                        help='Path to class weights file')
    
    # Data augmentation
    parser.add_argument('--hflip_prob', type=float, default=0.5,
                        help='Probability of horizontal flip')
    parser.add_argument('--color_jitter', type=float, nargs=4, default=[0.2, 0.2, 0.2, 0.1],
                        help='Color jitter parameters (brightness, contrast, saturation, hue)')
    
    # Optimizer
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd'],
                        help='Optimizer type')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for SGD')
    
    # Learning rate scheduler
    parser.add_argument('--scheduler', type=str, default='plateau',
                        choices=['step', 'plateau', 'none'],
                        help='Learning rate scheduler type')
    parser.add_argument('--step_size', type=int, default=10,
                        help='Step size for StepLR')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='Gamma for learning rate scheduler')
    parser.add_argument('--patience', type=int, default=3,
                        help='Patience for ReduceLROnPlateau')
    
    # Training
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of epochs to train')
    parser.add_argument('--clip_grad', type=float, default=0.0,
                        help='Gradient clipping value (0 = no clipping)')

    args = parser.parse_args()
    
    # Print arguments
    print("="*50)
    print("Training Configuration")
    print("="*50)
    for arg, value in vars(args).items():
        print(f"{arg:20}: {value}")
    print("="*50)
    
    train(args)