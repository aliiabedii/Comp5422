from .models import FCN_MT
from .utils import load_dense_data, ConfusionMatrix, DepthError
from . import dense_transforms
import torch
import numpy as np
import os


def test(args):
    from os import path
    
    # ============================================
    # 1. Setup Device
    # ============================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ============================================
    # 2. Load Model
    # ============================================
    model = FCN_MT(num_classes=19).to(device)
    
    # Try multiple possible paths for model file
    possible_paths = [
        args.checkpoint,
        'fcn_mt.th',
        path.join('homework', 'fcn_mt.th'),
        path.join('logs', 'fcn_mt', 'fcn_mt.th')
    ]
    
    checkpoint_path = None
    for p in possible_paths:
        if os.path.exists(p):
            checkpoint_path = p
            break
    
    if checkpoint_path is None:
        print("❌ No checkpoint found! Please train the model first.")
        print("Tried:", possible_paths)
        return
    
    print(f"✅ Loading model from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print("✅ Model loaded successfully!")
    
    # ============================================
    # 3. Setup Data Transforms (NO AUGMENTATION)
    # ============================================
    transform = dense_transforms.ToTensor3()
    
    # ============================================
    # 4. Load Data
    # ============================================
    print("\n--- Loading Datasets ---")
    
    train_loader = load_dense_data(
        args.train_dir,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        transform=transform
    )
    
    val_loader = load_dense_data(
        args.val_dir,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        transform=transform
    )
    
    print(f"Train loader: {len(train_loader)} batches ({len(train_loader.dataset)} samples)")
    print(f"Val loader: {len(val_loader)} batches ({len(val_loader.dataset)} samples)")
    
    # ============================================
    # 5. Evaluation Function
    # ============================================
    def evaluate_multitask(loader, name):
        print(f"\n{'='*50}")
        print(f"Evaluating on {name} set...")
        print(f"{'='*50}")
        
        # Segmentation metrics
        seg_confusion = ConfusionMatrix(size=19)
        
        # Depth metrics storage
        all_rel = []
        all_a1 = []
        all_a2 = []
        all_a3 = []
        
        with torch.no_grad():
            for batch_idx, (images, semantics, depths) in enumerate(loader):
                images = images.to(device)
                semantics = semantics.to(device)
                depths = depths.to(device)
                
                # Forward pass
                seg_logits, depth_pred = model(images)
                
                # === SEGMENTATION METRICS ===
                seg_pred = seg_logits.argmax(1)
                seg_confusion.add(seg_pred, semantics)
                
                # === DEPTH METRICS (mask invalid values) ===
                valid_mask = depths > 0  # Mask out invalid depth values
                
                if valid_mask.any():
                    # Get valid pixels only
                    depth_pred_valid = depth_pred.squeeze(1)[valid_mask].cpu().numpy()
                    depth_gt_valid = depths[valid_mask].cpu().numpy()
                    
                    # Compute depth errors using DepthError class
                    depth_error = DepthError(depth_gt_valid, depth_pred_valid)
                    rel, a1, a2, a3 = depth_error.compute_errors
                    
                    all_rel.append(rel)
                    all_a1.append(a1)
                    all_a2.append(a2)
                    all_a3.append(a3)
                
                # Progress
                if (batch_idx + 1) % 10 == 0:
                    print(f"  Processed {batch_idx+1}/{len(loader)} batches...")
        
        # === AGGREGATE SEGMENTATION METRICS ===
        seg_acc = seg_confusion.global_accuracy.item()
        seg_miou = seg_confusion.iou.item()
        per_class_iou = seg_confusion.class_iou.cpu().numpy()
        
        # === AGGREGATE DEPTH METRICS ===
        if all_rel:
            depth_rel = np.mean(all_rel)
            depth_a1 = np.mean(all_a1)
            depth_a2 = np.mean(all_a2)
            depth_a3 = np.mean(all_a3)
        else:
            depth_rel = depth_a1 = depth_a2 = depth_a3 = 0
            print("⚠️  No valid depth pixels found!")
        
        # Print results for this set
        print(f"\n📊 {name} Results:")
        print(f"  Segmentation - Accuracy: {seg_acc*100:.2f}%, mIoU: {seg_miou*100:.2f}%")
        print(f"  Depth - rel: {depth_rel:.4f}, δ<1.25: {depth_a1*100:.2f}%, δ<1.25²: {depth_a2*100:.2f}%, δ<1.25³: {depth_a3*100:.2f}%")
        
        return {
            'seg_accuracy': seg_acc,
            'seg_miou': seg_miou,
            'per_class_iou': per_class_iou,
            'depth_rel': depth_rel,
            'depth_a1': depth_a1,
            'depth_a2': depth_a2,
            'depth_a3': depth_a3
        }
    
    # ============================================
    # 6. Run Evaluation
    # ============================================
    train_metrics = evaluate_multitask(train_loader, "TRAIN")
    val_metrics = evaluate_multitask(val_loader, "VALIDATION")
    
    # ============================================
    # 7. Print Table 3
    # ============================================
    print("\n" + "="*100)
    print("TABLE 3: Multi-Task FCN Performance")
    print("="*100)
    print(f"{'Dataset':<12} {'Segmentation':^30} {'Depth':^50}")
    print(f"{'':<12} {'Accuracy':<10} {'mIoU':<10} {'rel':<12} {'δ<1.25':<12} {'δ<1.25²':<12} {'δ<1.25³':<12}")
    print("-"*100)
    
    # Train row
    print(f"{'Train':<12} "
          f"{train_metrics['seg_accuracy']*100:<10.2f}% "
          f"{train_metrics['seg_miou']*100:<10.2f}% "
          f"{train_metrics['depth_rel']:<12.4f} "
          f"{train_metrics['depth_a1']*100:<12.2f}% "
          f"{train_metrics['depth_a2']*100:<12.2f}% "
          f"{train_metrics['depth_a3']*100:<12.2f}%")
    
    # Validation row
    print(f"{'Validation':<12} "
          f"{val_metrics['seg_accuracy']*100:<10.2f}% "
          f"{val_metrics['seg_miou']*100:<10.2f}% "
          f"{val_metrics['depth_rel']:<12.4f} "
          f"{val_metrics['depth_a1']*100:<12.2f}% "
          f"{val_metrics['depth_a2']*100:<12.2f}% "
          f"{val_metrics['depth_a3']*100:<12.2f}%")
    print("="*100)
    
    # ============================================
    # 8. Optional: Print Per-class IoU
    # ============================================
    class_names = [
        'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 
        'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
        'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
    ]
    
    print("\n📊 Per-class IoU (Validation):")
    for i, (name, iou) in enumerate(zip(class_names, val_metrics['per_class_iou'])):
        print(f"  {i:2d} {name:<15}: {iou*100:5.2f}%")
    
    # ============================================
    # 9. Save Results to File
    # ============================================
    results_path = 'multitask_results.txt'
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write("Table 3: Multi-Task FCN Performance\n")
        f.write("="*100 + "\n")
        f.write(f"{'Dataset':<12} {'Segmentation':^30} {'Depth':^50}\n")
        f.write(f"{'':<12} {'Accuracy':<10} {'mIoU':<10} {'rel':<12} {'δ<1.25':<12} {'δ<1.25²':<12} {'δ<1.25³':<12}\n")
        f.write("-"*100 + "\n")
        
        f.write(f"{'Train':<12} "
                f"{train_metrics['seg_accuracy']*100:<10.2f}% "
                f"{train_metrics['seg_miou']*100:<10.2f}% "
                f"{train_metrics['depth_rel']:<12.4f} "
                f"{train_metrics['depth_a1']*100:<12.2f}% "
                f"{train_metrics['depth_a2']*100:<12.2f}% "
                f"{train_metrics['depth_a3']*100:<12.2f}%\n")
        
        f.write(f"{'Validation':<12} "
                f"{val_metrics['seg_accuracy']*100:<10.2f}% "
                f"{val_metrics['seg_miou']*100:<10.2f}% "
                f"{val_metrics['depth_rel']:<12.4f} "
                f"{val_metrics['depth_a1']*100:<12.2f}% "
                f"{val_metrics['depth_a2']*100:<12.2f}% "
                f"{val_metrics['depth_a3']*100:<12.2f}%\n")
        f.write("="*100 + "\n")
        
        f.write("\nPer-class IoU (Validation):\n")
        for i, (name, iou) in enumerate(zip(class_names, val_metrics['per_class_iou'])):
            f.write(f"  {i:2d} {name:<15}: {iou*100:5.2f}%\n")
    
    print(f"\n✅ Results saved to '{results_path}'")
    
    return val_metrics


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    
    # Required arguments
    parser.add_argument('--train_dir', type=str, required=True,
                        help='Path to training data')
    parser.add_argument('--val_dir', type=str, required=True,
                        help='Path to validation data')
    
    # Optional arguments
    parser.add_argument('--checkpoint', type=str, default='fcn_mt.th',
                        help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    args = parser.parse_args()
    
    # Run evaluation
    metrics = test(args)