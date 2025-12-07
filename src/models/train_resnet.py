"""
ResNet50 Model Training for Skin Cancer Classification.

This script trains a pretrained ResNet50 model with transfer learning
on the HAM10000 dataset. It handles class imbalance using weighted loss.
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import ResNet50_Weights
from tqdm import tqdm


# Class names for HAM10000
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']


def setup_device() -> torch.device:
    """
    Setup and return the computation device (GPU if available).
    
    Returns:
        torch.device: CUDA device if available, else CPU.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        print("⚠️  GPU not available, using CPU")
    
    return device


def get_data_transforms() -> Dict[str, transforms.Compose]:
    """
    Get data transforms for training and validation.
    
    Returns:
        Dictionary with 'train' and 'val' transforms.
    """
    # ImageNet normalization values
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std)
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std)
        ])
    }
    
    return data_transforms


def create_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[Dict[str, DataLoader], Dict[str, datasets.ImageFolder]]:
    """
    Create data loaders for training and validation.
    
    Args:
        data_dir: Base directory containing train/val folders.
        batch_size: Batch size for training.
        num_workers: Number of worker processes for data loading.
        
    Returns:
        Tuple of (dataloaders dict, datasets dict).
    """
    data_transforms = get_data_transforms()
    
    image_datasets = {
        'train': datasets.ImageFolder(
            os.path.join(data_dir, 'train'),
            data_transforms['train']
        ),
        'val': datasets.ImageFolder(
            os.path.join(data_dir, 'val'),
            data_transforms['val']
        )
    }
    
    dataloaders = {
        'train': DataLoader(
            image_datasets['train'],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        ),
        'val': DataLoader(
            image_datasets['val'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    }
    
    # Print dataset info
    print(f"\n📊 Dataset Info:")
    print(f"   Training samples: {len(image_datasets['train']):,}")
    print(f"   Validation samples: {len(image_datasets['val']):,}")
    print(f"   Classes: {image_datasets['train'].classes}")
    
    return dataloaders, image_datasets


def calculate_class_weights(dataset: datasets.ImageFolder) -> torch.Tensor:
    """
    Calculate class weights for handling class imbalance.
    
    Uses inverse frequency weighting: weight = total_samples / (num_classes * class_count)
    
    Args:
        dataset: ImageFolder dataset.
        
    Returns:
        Tensor of class weights.
    """
    # Count samples per class
    targets = np.array(dataset.targets)
    class_counts = np.bincount(targets, minlength=len(CLASS_NAMES))
    
    # Calculate inverse frequency weights
    total_samples = len(targets)
    num_classes = len(CLASS_NAMES)
    
    # Avoid division by zero
    class_counts = np.maximum(class_counts, 1)
    
    weights = total_samples / (num_classes * class_counts)
    
    print(f"\n⚖️  Class Weights (for imbalance correction):")
    for name, count, weight in zip(CLASS_NAMES, class_counts, weights):
        print(f"   {name}: {count:>5} samples → weight = {weight:.4f}")
    
    return torch.FloatTensor(weights)


def create_model(num_classes: int = 7, freeze_features: bool = True) -> nn.Module:
    """
    Create a ResNet50 model with a modified final layer.
    
    Args:
        num_classes: Number of output classes.
        freeze_features: Whether to freeze the feature extraction layers.
        
    Returns:
        Modified ResNet50 model.
    """
    print(f"\n🏗️  Building Model:")
    print(f"   Loading pretrained ResNet50...")
    
    # Load pretrained ResNet50
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    
    # Freeze feature extraction layers
    if freeze_features:
        print(f"   Freezing feature extraction layers...")
        for param in model.parameters():
            param.requires_grad = False
    
    # Replace the final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Final layer: {num_features} → 512 → {num_classes}")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    return model


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """
    Train for one epoch.
    
    Args:
        model: The model to train.
        dataloader: Training data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Computation device.
        
    Returns:
        Tuple of (average loss, accuracy).
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(dataloader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, float]:
    """
    Validate the model.
    
    Args:
        model: The model to validate.
        dataloader: Validation data loader.
        criterion: Loss function.
        device: Computation device.
        
    Returns:
        Tuple of (average loss, accuracy, f1_score).
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return epoch_loss, epoch_acc, epoch_f1


def plot_training_curves(
    history: Dict[str, List[float]],
    output_dir: str = "reports/figures"
):
    """
    Plot and save training curves.
    
    Args:
        history: Dictionary with training history.
        output_dir: Directory to save plots.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot 1: Loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    plt.title('ResNet50 Training: Loss Curves', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'loss_curve.png', dpi=150)
    plt.close()
    print(f"   📈 Saved: {output_path / 'loss_curve.png'}")
    
    # Plot 2: Accuracy curves
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    plt.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    plt.title('ResNet50 Training: Accuracy Curves', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path / 'accuracy_curve.png', dpi=150)
    plt.close()
    print(f"   📈 Saved: {output_path / 'accuracy_curve.png'}")
    
    # Plot 3: F1 Score curve
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['val_f1'], 'g-', label='Validation F1 Score', linewidth=2)
    plt.title('ResNet50 Training: F1 Score Curve', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('F1 Score (Weighted)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path / 'f1_curve.png', dpi=150)
    plt.close()
    print(f"   📈 Saved: {output_path / 'f1_curve.png'}")


def train_model(
    data_dir: str = "data/processed",
    model_dir: str = "models",
    num_epochs: int = 15,
    finetune_epochs: int = 15,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    finetune_lr: float = 1e-5,
    num_workers: int = 4
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Train the ResNet50 model with two phases:
    1. Feature extraction (frozen backbone)
    2. Fine-tuning (unfrozen layer4 + fc)
    
    Args:
        data_dir: Directory containing processed data.
        model_dir: Directory to save model checkpoints.
        num_epochs: Number of initial training epochs (frozen).
        finetune_epochs: Number of fine-tuning epochs.
        batch_size: Batch size.
        learning_rate: Learning rate for initial training.
        finetune_lr: Learning rate for fine-tuning (much lower).
        num_workers: Number of data loading workers.
        
    Returns:
        Tuple of (trained model, training history).
    """
    print("="*60)
    print("RESNET50 MODEL TRAINING")
    print("="*60)
    
    # Setup
    device = setup_device()
    
    # Create data loaders
    dataloaders, image_datasets = create_data_loaders(
        data_dir, batch_size, num_workers
    )
    
    # Calculate class weights
    class_weights = calculate_class_weights(image_datasets['train'])
    class_weights = class_weights.to(device)
    
    # Create model
    model = create_model(num_classes=7, freeze_features=True)
    model = model.to(device)
    
    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer (only optimize unfrozen parameters)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': []
    }
    
    # Create model directory
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    
    best_f1 = 0.0
    best_epoch = 0
    phase1_best_f1 = 0.0
    
    # ================================================================
    # PHASE 1: Feature Extraction (Frozen Backbone)
    # ================================================================
    print(f"\n🚀 PHASE 1: Feature Extraction (Frozen Backbone)")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Device: {device}")
    print("-"*60)
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, dataloaders['train'], criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc, val_f1 = validate(
            model, dataloaders['val'], criterion, device
        )
        
        # Update scheduler
        scheduler.step(val_f1)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        epoch_time = time.time() - epoch_start
        
        # Print epoch results
        print(f"Epoch {epoch+1:2d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"Val F1: {val_f1:.4f} | Time: {epoch_time:.1f}s")
        
        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch + 1
            best_model_path = model_path / "best_model_resnet.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_acc': val_acc,
                'class_names': CLASS_NAMES,
                'phase': 'feature_extraction'
            }, best_model_path)
            print(f"   💾 New best model saved! (F1: {val_f1:.4f})")
    
    phase1_best_f1 = best_f1
    phase1_best_epoch = best_epoch
    phase1_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"PHASE 1 COMPLETE - Feature Extraction")
    print(f"{'='*60}")
    print(f"   Best F1: {phase1_best_f1:.4f} (Epoch {phase1_best_epoch})")
    print(f"   Time: {phase1_time/60:.2f} minutes")
    
    # ================================================================
    # PHASE 2: Fine-Tuning (Unfreeze layer4 + fc)
    # ================================================================
    print(f"\n{'='*60}")
    print(f"🔓 PHASE 2: Fine-Tuning (Unfreezing layer4)")
    print(f"{'='*60}")
    
    # Unfreeze layer4 and fc
    print("\n   Unfreezing layers...")
    unfrozen_params = 0
    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True
            unfrozen_params += param.numel()
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Unfrozen parameters: {unfrozen_params:,}")
    print(f"   Total trainable parameters: {trainable_params:,}")
    
    # New optimizer with lower learning rate
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=finetune_lr
    )
    
    # New scheduler for fine-tuning
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )
    
    print(f"\n   Fine-tuning epochs: {finetune_epochs}")
    print(f"   Learning rate: {finetune_lr}")
    print("-"*60)
    
    finetune_start = time.time()
    
    for epoch in range(finetune_epochs):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, dataloaders['train'], criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc, val_f1 = validate(
            model, dataloaders['val'], criterion, device
        )
        
        # Update scheduler
        scheduler.step(val_f1)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        epoch_time = time.time() - epoch_start
        total_epoch = num_epochs + epoch + 1
        
        # Print epoch results
        print(f"Epoch {total_epoch:2d}/{num_epochs + finetune_epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"Val F1: {val_f1:.4f} | Time: {epoch_time:.1f}s")
        
        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = total_epoch
            best_model_path = model_path / "best_model_resnet.pth"
            torch.save({
                'epoch': total_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_acc': val_acc,
                'class_names': CLASS_NAMES,
                'phase': 'fine_tuning'
            }, best_model_path)
            print(f"   💾 New best model saved! (F1: {val_f1:.4f})")
    
    finetune_time = time.time() - finetune_start
    
    total_time = time.time() - start_time
    
    # Final summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"\n📊 Phase 1 Results (Feature Extraction):")
    print(f"   Best F1: {phase1_best_f1:.4f} (Epoch {phase1_best_epoch})")
    print(f"   Time: {phase1_time/60:.2f} minutes")
    
    print(f"\n📊 Phase 2 Results (Fine-Tuning):")
    print(f"   Best F1: {best_f1:.4f} (Epoch {best_epoch})")
    print(f"   Time: {finetune_time/60:.2f} minutes")
    
    print(f"\n📊 Overall Results:")
    print(f"   Best Validation F1: {best_f1:.4f} (Epoch {best_epoch})")
    print(f"   Best Validation Acc: {history['val_acc'][best_epoch-1]:.4f}")
    print(f"   Improvement from Phase 1: +{(best_f1 - phase1_best_f1)*100:.2f}% F1")
    print(f"   Total training time: {total_time/60:.2f} minutes")
    print(f"   Model saved to: {model_path / 'best_model_resnet.pth'}")
    
    # Plot training curves
    print(f"\n📈 Generating training plots...")
    plot_training_curves(history)
    
    # Summary line for comparison
    print("\n" + "="*60)
    print(f"[RESNET50] Deep Learning - Val Acc: {history['val_acc'][best_epoch-1]:.2f}, "
          f"F1: {best_f1:.2f}")
    print("="*60)
    
    return model, history


if __name__ == "__main__":
    # Get project root directory
    project_root = Path(__file__).parent.parent.parent
    os.chdir(project_root)
    
    print(f"Working directory: {os.getcwd()}")
    
    # Run training with fine-tuning
    model, history = train_model(
        num_epochs=15,        # Phase 1: Feature extraction
        finetune_epochs=15,   # Phase 2: Fine-tuning
        batch_size=32,
        learning_rate=0.001,
        finetune_lr=1e-5,     # Lower LR for fine-tuning
        num_workers=2         # Reduce workers for stability
    )
    
    print("\n✅ ResNet50 training complete!")
