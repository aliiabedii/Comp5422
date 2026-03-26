import torch
import numpy as np

from .models import FCN_MT, save_model
from .utils import load_dense_data, ConfusionMatrix, load_class_weights, DepthError
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
    model = FCN_MT(num_classes=19).to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Setup TensorBoard loggers
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)
        print(f"Logging to: {args.log_dir}")

    # ============================================
    # 1. Load Class Weights
    # ============================================
    class_weights = load_class_weights(args.weight_file).to(device)
    print(f"Class weights loaded: {class_weights.shape}")
    
    # ============================================
    # 2. Setup Data Loaders with Augmentation
    # ============================================
    # Training transforms with augmentation
    train_transform = dense_transforms.Compose3([
        dense_transforms.RandomHorizontalFlip3(flip_prob=args.hflip_prob),
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
    # 3. Load Pretrained Single-Task Weights (Optional)
    # ============================================
    if args.pretrained_st is not None and os.path.exists(args.pretrained_st):
        print(f"Loading pretrained single-task weights from {args.pretrained_st}")
        st_dict = torch.load(args.pretrained_st, map_location=device)
        mt_dict = model.state_dict()
        
        # Filter out mismatched keys (depth head doesn't exist in single-task)
        pretrained_dict = {k: v for k, v in st_dict.items() 
                          if k in mt_dict and 'depth' not in k}
        
        mt_dict.update(pretrained_dict)
        model.load_state_dict(mt_dict)
        print(f"✅ Loaded {len(pretrained_dict)} layers from single-task model")
    
    # ============================================
    # 4. Setup Loss Functions
    # ============================================
    # Segmentation loss (with class weights)
    seg_criterion = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
    
    # Depth loss (L1 for regression)
    depth_criterion = torch.nn.L1Loss()
    
    # ============================================
    # 5. Setup Optimizer and Scheduler
    # ============================================
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=args.gamma, patience=args.patience
    )
    
    # ============================================
    # 6. Training Loop
    # ============================================
    num_epochs = args.num_epochs
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*50}")
        
        # --- Training Phase ---
        model.train()
        train_seg_loss = 0.0
        train_depth_loss = 0.0
        train_confusion = ConfusionMatrix(size=19)
        
        for batch_idx, (images, semantics, depths) in enumerate(train_loader):
            images = images.to(device)
            semantics = semantics.to(device)
            depths = depths.to(device)
            
            # Forward pass (multi-task returns two outputs)
            seg_logits, depth_pred = model(images)
            
            # Segmentation loss
            seg_loss = seg_criterion(seg_logits, semantics)
            
            # Depth loss (mask invalid values)
            valid_mask = depths > 0  # Shape: [B, H, W]
            if valid_mask.any():
                # Flatten both tensors to 1D for loss computation
                depth_pred_flat = depth_pred.squeeze(1)[valid_mask]  # Remove channel dim, then mask
                depths_flat = depths[valid_mask]
                depth_loss = depth_criterion(depth_pred_flat, depths_flat)
            else:
                depth_loss = torch.tensor(0.0, device=device)
            
            # Combined loss (weighted sum)
            total_loss = seg_loss + args.depth_weight * depth_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping (optional)
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            
            optimizer.step()
            
            # Update metrics
            train_seg_loss += seg_loss.item()
            train_depth_loss += depth_loss.item() if depth_loss > 0 else 0
            train_confusion.add(seg_logits.argmax(1), semantics)
            
            # Log training losses
            if train_logger is not None and global_step % args.log_freq == 0:
                train_logger.add_scalar('loss/segmentation', seg_loss.item(), global_step)
                train_logger.add_scalar('loss/depth', depth_loss.item(), global_step)
                train_logger.add_scalar('loss/total', total_loss.item(), global_step)
                
                # Log images every 500 steps
                if global_step % 500 == 0:
                    log(train_logger, images, semantics, seg_logits, global_step)
            
            global_step += 1
            
            if batch_idx % args.print_freq == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)} | "
                      f"Seg Loss: {seg_loss.item():.4f} | "
                      f"Depth Loss: {depth_loss.item():.4f}")
        
        # Calculate training epoch metrics
        train_seg_loss /= len(train_loader)
        train_depth_loss /= len(train_loader)
        train_acc = train_confusion.global_accuracy.item()
        train_miou = train_confusion.iou.item()
        
        print(f"\nTrain - Seg Loss: {train_seg_loss:.4f}, Depth Loss: {train_depth_loss:.4f}")
        print(f"Train - Acc: {train_acc:.4f}, mIoU: {train_miou:.4f}")
        
        # Log training epoch metrics
        if train_logger is not None:
            train_logger.add_scalar('epoch/train_seg_loss', train_seg_loss, epoch)
            train_logger.add_scalar('epoch/train_depth_loss', train_depth_loss, epoch)
            train_logger.add_scalar('epoch/train_acc', train_acc, epoch)
            train_logger.add_scalar('epoch/train_miou', train_miou, epoch)
        
        # --- Validation Phase ---
        model.eval()
        val_seg_loss = 0.0
        val_depth_loss = 0.0
        val_confusion = ConfusionMatrix(size=19)
        val_depth_errors = []
        
        with torch.no_grad():
            for images, semantics, depths in val_loader:
                images = images.to(device)
                semantics = semantics.to(device)
                depths = depths.to(device)
                
                seg_logits, depth_pred = model(images)
                
                # Segmentation loss
                seg_loss = seg_criterion(seg_logits, semantics)
                val_seg_loss += seg_loss.item()
                val_confusion.add(seg_logits.argmax(1), semantics)
                
                # Depth metrics
                valid_mask = depths > 0
                if valid_mask.any():
                    depth_loss = depth_criterion(depth_pred.squeeze(1)[valid_mask], depths[valid_mask])
                    val_depth_loss += depth_loss.item()
                    
                    # For DepthError, we need numpy arrays
                    depth_error = DepthError(
                        depths[valid_mask].cpu().numpy(),
                        depth_pred.squeeze(1)[valid_mask].cpu().numpy()  # Squeeze and mask
                    )
                    rel, a1, a2, a3 = depth_error.compute_errors
                    val_depth_errors.append([rel, a1, a2, a3])
        
        # Calculate validation metrics
        val_seg_loss /= len(val_loader)
        val_depth_loss /= len(val_loader)
        val_acc = val_confusion.global_accuracy.item()
        val_miou = val_confusion.iou.item()
        
        # Aggregate depth metrics
        if val_depth_errors:
            depth_avg = np.mean(val_depth_errors, axis=0)
            val_rel, val_a1, val_a2, val_a3 = depth_avg
        else:
            val_rel = val_a1 = val_a2 = val_a3 = 0
        
        print(f"\nVal - Seg Loss: {val_seg_loss:.4f}, Depth Loss: {val_depth_loss:.4f}")
        print(f"Val - Acc: {val_acc:.4f}, mIoU: {val_miou:.4f}")
        print(f"Val Depth - rel: {val_rel:.4f}, a1: {val_a1:.4f}, a2: {val_a2:.4f}, a3: {val_a3:.4f}")
        
        # Log validation metrics
        if valid_logger is not None:
            valid_logger.add_scalar('epoch/val_seg_loss', val_seg_loss, epoch)
            valid_logger.add_scalar('epoch/val_depth_loss', val_depth_loss, epoch)
            valid_logger.add_scalar('epoch/val_acc', val_acc, epoch)
            valid_logger.add_scalar('epoch/val_miou', val_miou, epoch)
            valid_logger.add_scalar('epoch/val_rel', val_rel, epoch)
            valid_logger.add_scalar('epoch/val_a1', val_a1, epoch)
            valid_logger.add_scalar('epoch/val_a2', val_a2, epoch)
            valid_logger.add_scalar('epoch/val_a3', val_a3, epoch)
        
        # Learning rate scheduler step (based on combined validation loss)
        val_total_loss = val_seg_loss + args.depth_weight * val_depth_loss
        scheduler.step(val_total_loss)
        
        # Save best model based on combined validation loss
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            save_model(model)
            print(f"✅ Model saved! Best val loss: {best_val_loss:.4f}")
    
    print(f"\n{'='*50}")
    print(f"Training Complete!")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"{'='*50}")
    
    save_model(model)
    return best_val_loss


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
    
    # Required paths
    parser.add_argument('--train_dir', type=str, required=True,
                        help='Path to training data')
    parser.add_argument('--val_dir', type=str, required=True,
                        help='Path to validation data')
    
    # Optional paths
    parser.add_argument('--pretrained_st', type=str, default=None,
                        help='Path to pretrained single-task model (optional)')
    
    # Logging
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Directory for TensorBoard logs')
    parser.add_argument('--log_freq', type=int, default=10)
    parser.add_argument('--print_freq', type=int, default=10)
    
    # Data
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--weight_file', type=str, default='classweight.cls')
    
    # Data augmentation
    parser.add_argument('--hflip_prob', type=float, default=0.5)
    parser.add_argument('--color_jitter', type=float, nargs=4, default=[0.2, 0.2, 0.2, 0.1])
    
    # Optimization
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--depth_weight', type=float, default=1.0)
    parser.add_argument('--clip_grad', type=float, default=1.0)
    
    # LR scheduler
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--patience', type=int, default=3)
    
    # Training
    parser.add_argument('--num_epochs', type=int, default=50)

    args = parser.parse_args()
    train(args)