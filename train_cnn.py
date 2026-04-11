from .models import CNNClassifier, save_model, SoftmaxCrossEntropyLoss
from .utils import ConfusionMatrix, load_data, VehicleClassificationDataset
import torch
import torchvision
import torch.utils.tensorboard as tb
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import os


def train(args):
    from os import path
    
    # ============================================
    # 1. Setup Device
    # ============================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ============================================
    # 2. Initialize Model
    # ============================================
    model = CNNClassifier(num_classes=6)
    model = model.to(device)
    
    # ============================================
    # 3. Setup TensorBoard Logging (Section 3.3)
    # ============================================
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)
        print(f"Logging to: {args.log_dir}")
    
    # ============================================
    # 4. Setup Data Loaders
    # ============================================
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
    ])
    
    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
    ])
    
    # Load datasets
    train_dataset = VehicleClassificationDataset(args.train_dir, transform=train_transform)
    val_dataset = VehicleClassificationDataset(args.val_dir, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                            shuffle=False, num_workers=0, drop_last=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # ============================================
    # 5. Setup Loss, Optimizer, and Scheduler
    # ============================================
    criterion = SoftmaxCrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), 
                           lr=args.lr,
                           weight_decay=args.weight_decay)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # ============================================
    # 6. Training Loop (Section 3.4)
    # ============================================
    num_epochs = args.num_epochs
    global_step = 0
    best_val_acc = 0.0
    patience = 15
    patience_counter = 0
    
    for epoch in range(num_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*50}")
        
        # --- Training Phase ---
        model.train()
        
        # ✅ CRITICAL: Initialize these variables BEFORE the loop
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # ✅ Accumulate statistics
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(logits, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
            
            # Log training loss at every iteration (Section 3.3)
            if train_logger is not None:
                train_logger.add_scalar('train/loss', loss.item(), global_step)
            
            global_step += 1
        
        # Calculate training accuracy
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        
        # Log training accuracy at each epoch
        if train_logger is not None:
            train_logger.add_scalar('train/accuracy', train_acc, epoch)
        
        # --- Validation Phase ---
        model.eval()
        
        # ✅ CRITICAL: Initialize these variables BEFORE validation loop
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                logits = model(images)
                loss = criterion(logits, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(logits, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        
        # Calculate validation accuracy
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Log validation accuracy at each epoch
        if valid_logger is not None:
            valid_logger.add_scalar('valid/accuracy', val_acc, epoch)
            valid_logger.add_scalar('valid/loss', val_loss, epoch)
        
        # Learning rate scheduler step
        scheduler.step()
        
        # --- Save Best Model ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model)  # Saves as cnn.pth
            print(f"✅ Model saved! Best Val Acc: {best_val_acc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{patience}")
        
        # --- Early Stopping ---
        if patience_counter >= patience:
            print(f"⚠️  Early stopping at epoch {epoch+1}")
            break
    
    print(f"\n{'='*50}")
    print(f"Training Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"{'='*50}")
    
    # Close TensorBoard writers
    if train_logger is not None:
        train_logger.close()
    if valid_logger is not None:
        valid_logger.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    
    # Required arguments
    parser.add_argument('--log_dir', type=str, default='logs/cnn', 
                        help='Directory for TensorBoard logs')
    parser.add_argument('--train_dir', type=str, 
                        default=r'C:\Users\Lenovo\Desktop\computer Vision - COMP 5422\HW1_Dataset\train_subset',
                        help='Path to training dataset')
    parser.add_argument('--val_dir', type=str, 
                        default=r'C:\Users\Lenovo\Desktop\computer Vision - COMP 5422\HW1_Dataset\validation_subset',
                        help='Path to validation dataset')
    
    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='L2 regularization')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of epochs')
    
    args = parser.parse_args()
    train(args)