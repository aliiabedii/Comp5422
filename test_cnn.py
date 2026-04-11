from homework.models import CNNClassifier
from homework.utils import ConfusionMatrix, load_data
import torch
import torchvision
import torch.utils.tensorboard as tb
from torchvision import transforms
import os


def test(args):
    from os import path
    
    # ============================================
    # 1. Setup Device
    # ============================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ============================================
    # 2. Load Model (EVALUATION ONLY - NO TRAINING)
    # ============================================
    model = CNNClassifier(num_classes=6)
    
    # Try both possible extensions
    checkpoint_path = None
    for fname in ['cnn.th', 'cnn.pth']:
        if os.path.exists(fname):
            checkpoint_path = fname
            print(f"✅ Found checkpoint: {fname}")
            break
    
    if checkpoint_path is None:
        print("❌ No checkpoint found! Please train the model first.")
        return 0.0
    
    # Load the saved weights
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()  # IMPORTANT: Set to evaluation mode
    print(f"✅ Model loaded from {checkpoint_path}")
    
    # ============================================
    # 3. Setup Data Transforms (NO AUGMENTATION)
    # ============================================
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # ============================================
    # 4. Load Data
    # ============================================
    print("\n--- Loading Datasets ---")
    
    train_loader = load_data(args.train_dir, 
                            batch_size=args.batch_size, 
                            num_workers=4,
                            transform=transform)
    
    val_loader = load_data(args.val_dir,
                          batch_size=args.batch_size,
                          num_workers=4,
                          transform=transform)
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # ============================================
    # 5. Evaluation Function (NO GRADIENTS)
    # ============================================
    def evaluate(loader, name):
        confusion = ConfusionMatrix(size=6)
        correct = 0
        total = 0
        
        with torch.no_grad():  # IMPORTANT: No gradients needed for evaluation
            for i, (images, labels) in enumerate(loader):
                images = images.to(device)
                labels = labels.to(device)
                
                # Forward pass only
                logits = model(images)
                preds = logits.argmax(1)
                
                # Update metrics
                confusion.add(preds, labels)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
                if (i+1) % 10 == 0:
                    print(f"  Processed {i+1}/{len(loader)} batches...")
        
        # Calculate metrics
        accuracy = correct / total
        class_acc = confusion.class_accuracy
        iou = confusion.iou
        
        print(f"\n📊 {name} Set Results:")
        print(f"  Global Accuracy: {accuracy:.4f}")
        print(f"  Mean IoU: {iou:.4f}")
        
        return accuracy, iou, confusion
    
    # ============================================
    # 6. Run Evaluation
    # ============================================
    print("\n" + "="*50)
    print("Evaluating on Training Set...")
    print("="*50)
    train_acc, train_iou, train_confusion = evaluate(train_loader, "Training")
    
    print("\n" + "="*50)
    print("Evaluating on Validation Set...")
    print("="*50)
    val_acc, val_iou, val_confusion = evaluate(val_loader, "Validation")
    
    # ============================================
    # 7. Print Summary Table (Table 1 format)
    # ============================================
    print("\n" + "="*50)
    print("TABLE 1: Classification Performance")
    print("="*50)
    print(f"{'Dataset':<15} {'Accuracy':<15}")
    print("-"*30)
    print(f"{'Training':<15} {train_acc:<15.4f}")
    print(f"{'Validation':<15} {val_acc:<15.4f}")
    print("="*50)
    
    # ============================================
    # 8. Save results to file
    # ============================================
    results_path = 'cls_results.txt'
    with open(results_path, 'w') as f:
        f.write("Table 1: Classification Performance on VehicleClassification Dataset\n")
        f.write("="*50 + "\n")
        f.write(f"{'Dataset':<15} {'Accuracy':<15}\n")
        f.write("-"*30 + "\n")
        f.write(f"{'Training':<15} {train_acc:<15.4f}\n")
        f.write(f"{'Validation':<15} {val_acc:<15.4f}\n")
        f.write("="*50 + "\n")
        f.write(f"\nModel: {checkpoint_path}\n")
        f.write(f"Model download link: [INSERT YOUR LINK HERE]\n")
    
    print(f"\n✅ Results saved to {results_path}")
    print("\nDon't forget to:")
    print("1. Add your TensorBoard training curves to cls_results.pdf")
    print("2. Upload your model and add download link")
    
    return val_acc,train_confusion, val_confusion


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    
    # Required arguments
    parser.add_argument('--train_dir', type=str, 
                        default=r'C:\Users\Lenovo\Desktop\computer Vision - COMP 5422\HW1_Dataset\train_subset',
                        help='Path to training dataset')
    parser.add_argument('--val_dir', type=str, 
                        default=r'C:\Users\Lenovo\Desktop\computer Vision - COMP 5422\HW1_Dataset\validation_subset',
                        help='Path to validation dataset')
    
    # Optional arguments
    parser.add_argument('--log_dir', type=str, default='logs/eval',
                        help='Directory for logs (optional)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for evaluation')

    args = parser.parse_args()
    
    # Run evaluation
    accuracy, confusion_matrix = test(args)
    print(f"\nFinal Validation Accuracy: {accuracy:.4f}")
    print("Confusion matrix available as 'confusion_matrix'")