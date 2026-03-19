import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from .models import FCN_MT 
from .utils import load_kitti_data
from . import dense_transforms
import torch.utils.tensorboard as tb


def test_kitti(args):  # Note: function name is test_kitti, not test
    from os import path
    """
    Your code here
    Hint: load the saved checkpoint of your model, and perform evaluation for both segmentation and depth estimation tasks 
          on the provided images of the KITTI dataset
    """
    
    # ============================================
    # 1. Setup Device
    # ============================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ============================================
    # 2. Load Multi-Task Model
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
        print("❌ No checkpoint found! Please train the multi-task model first.")
        print("Tried:", possible_paths)
        return
    
    print(f"✅ Loading model from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print("✅ Model loaded successfully!")
    
    # ============================================
    # 3. Load KITTI Data
    # ============================================
    transform = dense_transforms.ToTensor3()
    
    kitti_loader = load_kitti_data(
        args.kitti_dir,
        batch_size=1,  # Process one image at a time for visualization
        shuffle=False,
        num_workers=args.num_workers,
        transform=transform
    )
    
    print(f"\nKITTI test samples: {len(kitti_loader.dataset)}")
    
    # ============================================
    # 4. Run Inference and Collect Results
    # ============================================
    all_images = []
    all_pred_seg = []
    all_pred_depth = []
    all_filenames = []
    
    print("\nRunning inference on KITTI samples...")
    
    with torch.no_grad():
        for batch_idx, (images, semantics, depths) in enumerate(kitti_loader):
            images = images.to(device)
            
            # Forward pass
            seg_logits, depth_pred = model(images)
            
            # Get predictions
            seg_pred = seg_logits.argmax(1).cpu().numpy()[0]
            depth_pred_np = depth_pred.cpu().numpy()[0, 0]  # Remove batch and channel dims
            
            # Store for visualization
            all_images.append(images.cpu().numpy()[0])
            all_pred_seg.append(seg_pred)
            all_pred_depth.append(depth_pred_np)
            
            # Try to get filename if available
            if hasattr(kitti_loader.dataset, 'image_files') and batch_idx < len(kitti_loader.dataset.image_files):
                filename = os.path.basename(kitti_loader.dataset.image_files[batch_idx])
                all_filenames.append(filename)
            
            if (batch_idx + 1) % 5 == 0:
                print(f"  Processed {batch_idx+1}/{len(kitti_loader)} images")
    
    print(f"✅ Completed inference on {len(all_images)} images")
    
    # ============================================
    # 5. Visualize Results (Figure 3 style)
    # ============================================
    print("\n📊 Creating visualizations...")
    
    # Select 6 random samples or first 6
    num_samples = min(6, len(all_images))
    if len(all_images) > 6:
        indices = np.random.choice(len(all_images), num_samples, replace=False)
    else:
        indices = range(num_samples)
    
    # Create figure with 5 rows and num_samples columns
    fig, axes = plt.subplots(5, num_samples, figsize=(num_samples*4, 20))
    
    # If only one sample, axes is 1D
    if num_samples == 1:
        axes = axes.reshape(5, 1)
    
    # Class names for reference
    class_names = [
        'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 
        'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
        'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
    ]
    
    for i, idx in enumerate(indices):
        # Get data
        img = all_images[idx]
        pred_seg = all_pred_seg[idx]
        pred_depth = all_pred_depth[idx]
        
        # Get filename if available
        filename = all_filenames[idx] if idx < len(all_filenames) else f"sample_{idx}"
        
        # Denormalize image
        img = img.transpose(1, 2, 0)  # CHW -> HWC
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1)
        
        # Row 1: Input Image
        axes[0, i].imshow(img)
        axes[0, i].set_title(f'{filename[:10]}...', fontsize=8)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Input Image', fontsize=12)
        
        # Row 2: Predicted Depth
        depth_im = axes[1, i].imshow(pred_depth, cmap='plasma')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('Pred Depth', fontsize=12)
        
        # Row 3: Ground Truth Depth (not available for KITTI)
        axes[2, i].text(0.5, 0.5, 'No GT', ha='center', va='center', fontsize=10)
        axes[2, i].set_facecolor('lightgray')
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_ylabel('GT Depth', fontsize=12)
        
        # Row 4: Predicted Semantic
        sem_im = axes[3, i].imshow(pred_seg, cmap='tab20', vmin=0, vmax=18)
        axes[3, i].axis('off')
        if i == 0:
            axes[3, i].set_ylabel('Pred Semantic', fontsize=12)
        
        # Row 5: Ground Truth Semantic (not available for KITTI)
        axes[4, i].text(0.5, 0.5, 'No GT', ha='center', va='center', fontsize=10)
        axes[4, i].set_facecolor('lightgray')
        axes[4, i].axis('off')
        if i == 0:
            axes[4, i].set_ylabel('GT Semantic', fontsize=12)
    
    # Add colorbars
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.05)
    cbar_ax_depth = fig.add_axes([0.15, 0.02, 0.7, 0.01])
    cbar_depth = fig.colorbar(depth_im, cax=cbar_ax_depth, orientation='horizontal')
    cbar_depth.set_label('Depth (m)')
    
    cbar_ax_sem = fig.add_axes([0.15, 0.00, 0.7, 0.01])
    cbar_sem = fig.colorbar(sem_im, cax=cbar_ax_sem, orientation='horizontal')
    cbar_sem.set_label('Semantic Class')
    
    plt.suptitle('KITTI Cross-Dataset Evaluation (Cityscapes Model)', fontsize=16, y=0.98)
    
    # Save figure
    os.makedirs('kitti_results', exist_ok=True)
    plt.savefig('kitti_results/kitti_evaluation.png', dpi=150, bbox_inches='tight')
    print(f"✅ Visualization saved to 'kitti_results/kitti_evaluation.png'")
    plt.show()
    
    # ============================================
    # 6. Analysis
    # ============================================
    print("\n" + "="*60)
    print("ANALYSIS: Cityscapes Model on KITTI Dataset")
    print("="*60)
    print("""
    Observations:
    - The model trained on Cityscapes is being applied to KITTI images
    - KITTI has different characteristics:
      * Camera perspective: Mounted on a car (lower angle) vs. Cityscapes (street view)
      * Scene composition: More road, cars; fewer buildings, pedestrians
      * Lighting conditions: Different weather/lighting
      * Resolution: May differ from Cityscapes 128x256
    
    Expected Challenges:
    1. Domain Shift: The model may struggle with KITTI's different appearance
    2. Class Distribution: KITTI has more road/car pixels, fewer building/sky
    3. Depth Range: KITTI depth statistics may differ from Cityscapes
    
    Qualitative Assessment:
    - Look at the predicted semantic maps:
      * Are roads correctly identified? (Class 0)
      * Are cars detected? (Class 13)
      * Are buildings/sky misclassified?
    
    - Look at depth predictions:
      * Does depth gradually increase with distance?
      * Are there artifacts or unrealistic depth values?
    
    Recommendations for Improvement:
    - Fine-tune on KITTI data if available
    - Use domain adaptation techniques
    - Apply dataset-specific normalization
    """)
    
    return all_images, all_pred_seg, all_pred_depth


# Keep the original test function for compatibility
def test(args):
    """Wrapper function for backward compatibility"""
    return test_kitti(args)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--kitti_dir', type=str, required=True,
                        help='Path to KITTI test samples')
    parser.add_argument('--checkpoint', type=str, default='fcn_mt.th',
                        help='Path to model checkpoint')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers')
    parser.add_argument('--log_dir', type=str, default='logs/kitti_eval',
                        help='Directory for logs (optional)')

    args = parser.parse_args()
    test_kitti(args)