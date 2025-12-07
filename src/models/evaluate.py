"""
Model Evaluation Script for Skin Cancer Classification.

This script evaluates the trained ResNet50 model on the held-out test set
and generates comprehensive metrics, visualizations, and explainability outputs.
"""

import os
import random
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import ResNet50_Weights
from tqdm import tqdm


# Class names for HAM10000
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
CLASS_FULL_NAMES = {
    'akiec': 'Actinic Keratoses',
    'bcc': 'Basal Cell Carcinoma',
    'bkl': 'Benign Keratosis',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic Nevi',
    'vasc': 'Vascular Lesions'
}


def setup_device() -> torch.device:
    """Setup computation device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("⚠️  Using CPU")
    return device


def load_model(model_path: str, device: torch.device) -> nn.Module:
    """
    Load the trained ResNet50 model.
    
    Args:
        model_path: Path to saved model checkpoint.
        device: Computation device.
        
    Returns:
        Loaded model in eval mode.
    """
    print(f"\n📂 Loading model from: {model_path}")
    
    # Create model architecture
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 7)
    )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"   Loaded from epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"   Training phase: {checkpoint.get('phase', 'N/A')}")
    print(f"   Validation F1: {checkpoint.get('val_f1', 'N/A'):.4f}")
    
    model = model.to(device)
    model.eval()
    
    return model


def get_test_dataloader(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 2
) -> Tuple[DataLoader, datasets.ImageFolder]:
    """
    Create test data loader.
    
    Args:
        data_dir: Directory containing test data.
        batch_size: Batch size.
        num_workers: Number of data loading workers.
        
    Returns:
        Tuple of (dataloader, dataset).
    """
    # ImageNet normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'test'),
        transform=transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\n📊 Test Dataset:")
    print(f"   Samples: {len(test_dataset):,}")
    print(f"   Classes: {test_dataset.classes}")
    
    return test_loader, test_dataset


def run_inference(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference on the test set.
    
    Args:
        model: Trained model.
        dataloader: Test data loader.
        device: Computation device.
        
    Returns:
        Tuple of (predictions, true_labels, probabilities).
    """
    print("\n🔮 Running inference on test set...")
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str = "reports/metrics_report.txt"
) -> dict:
    """
    Calculate and save evaluation metrics.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        output_path: Path to save metrics report.
        
    Returns:
        Dictionary with metrics.
    """
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(CLASS_NAMES))
    )
    
    # Classification report
    report = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        digits=4
    )
    
    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save metrics report
    with open(output_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("DERMAOPS MODEL EVALUATION REPORT\n")
        f.write("Test Set Performance - ResNet50 (Fine-tuned)\n")
        f.write("="*60 + "\n\n")
        
        f.write("OVERALL METRICS\n")
        f.write("-"*40 + "\n")
        f.write(f"Test Accuracy:     {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        f.write(f"F1 Score (Weighted): {f1_weighted:.4f}\n")
        f.write(f"F1 Score (Macro):    {f1_macro:.4f}\n\n")
        
        f.write("CLASSIFICATION REPORT\n")
        f.write("-"*40 + "\n")
        f.write(report)
        f.write("\n\n")
        
        f.write("PER-CLASS BREAKDOWN\n")
        f.write("-"*40 + "\n")
        f.write(f"{'Class':<8} {'Full Name':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}\n")
        f.write("-"*74 + "\n")
        for i, name in enumerate(CLASS_NAMES):
            full_name = CLASS_FULL_NAMES.get(name, name)
            f.write(f"{name:<8} {full_name:<25} {precision[i]:>10.4f} {recall[i]:>10.4f} {f1[i]:>10.4f} {support[i]:>10}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("Generated by DermaOps Evaluation Pipeline\n")
    
    print(f"\n✅ Metrics report saved to: {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SET EVALUATION RESULTS")
    print("="*60)
    print(f"\n📊 Overall Metrics:")
    print(f"   Test Accuracy:      {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   F1 Score (Weighted): {f1_weighted:.4f}")
    print(f"   F1 Score (Macro):    {f1_macro:.4f}")
    print(f"\n📋 Classification Report:")
    print(report)
    
    return {
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'precision': precision,
        'recall': recall,
        'f1_per_class': f1,
        'support': support
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str = "reports/figures/confusion_matrix_test.png"
):
    """
    Generate and save confusion matrix heatmap.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        output_path: Path to save figure.
    """
    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize for percentages (optional second plot)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Raw counts
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d',
        cmap='Blues',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=axes[0]
    )
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    axes[0].set_ylabel('True Label', fontsize=12)
    
    # Plot 2: Normalized (percentages)
    sns.heatmap(
        cm_normalized, 
        annot=True, 
        fmt='.2%',
        cmap='Blues',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=axes[1]
    )
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Predicted Label', fontsize=12)
    axes[1].set_ylabel('True Label', fontsize=12)
    
    plt.suptitle('ResNet50 Test Set Performance - Confusion Matrix', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Confusion matrix saved to: {output_path}")


class GradCAM:
    """Grad-CAM implementation for model explainability."""
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)
    
    def _forward_hook(self, module, input, output):
        self.activations = output.detach()
    
    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        """Generate Grad-CAM heatmap."""
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        target = output[0, target_class]
        target.backward()
        
        # Get gradients and activations
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2), keepdim=True)  # [C, 1, 1]
        
        # Weighted combination
        cam = (weights * activations).sum(dim=0)  # [H, W]
        
        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.cpu().numpy()


def generate_gradcam_samples(
    model: nn.Module,
    test_dataset: datasets.ImageFolder,
    device: torch.device,
    output_path: str = "reports/figures/explainability_samples.png",
    num_samples: int = 3,
    target_class: str = 'mel'
):
    """
    Generate Grad-CAM visualizations for sample images.
    
    Args:
        model: Trained model.
        test_dataset: Test dataset.
        device: Computation device.
        output_path: Path to save figure.
        num_samples: Number of samples to visualize.
        target_class: Class to visualize (default: melanoma).
    """
    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Find target class index
    class_idx = CLASS_NAMES.index(target_class)
    
    # Get samples from target class
    target_indices = [
        i for i, (_, label) in enumerate(test_dataset.samples)
        if label == class_idx
    ]
    
    if len(target_indices) < num_samples:
        print(f"⚠️  Only {len(target_indices)} samples available for class '{target_class}'")
        num_samples = len(target_indices)
    
    # Random sample selection
    selected_indices = random.sample(target_indices, num_samples)
    
    # Setup Grad-CAM
    target_layer = model.layer4[-1]  # Last block of layer4
    gradcam = GradCAM(model, target_layer)
    
    # Create figure
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    # Denormalization transform
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    print(f"\n🔬 Generating Grad-CAM for {num_samples} '{target_class}' samples...")
    
    for i, idx in enumerate(selected_indices):
        # Load image
        img_path, true_label = test_dataset.samples[idx]
        
        # Original image (for display)
        original_img = Image.open(img_path).convert('RGB')
        original_img = original_img.resize((224, 224))
        original_np = np.array(original_img)
        
        # Transformed image (for model)
        img_tensor, _ = test_dataset[idx]
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            output = model(img_tensor)
            probs = F.softmax(output, dim=1)
            pred_class = output.argmax(dim=1).item()
            pred_prob = probs[0, pred_class].item()
        
        # Generate Grad-CAM
        cam = gradcam.generate(img_tensor, pred_class)
        
        # Resize CAM to image size
        cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224))) / 255
        
        # Create heatmap overlay
        heatmap = plt.cm.jet(cam_resized)[:, :, :3]
        overlay = (0.6 * original_np / 255 + 0.4 * heatmap)
        overlay = np.clip(overlay, 0, 1)
        
        # Plot original image
        axes[i, 0].imshow(original_np)
        axes[i, 0].set_title(f'Original\nTrue: {CLASS_NAMES[true_label]}', fontsize=10)
        axes[i, 0].axis('off')
        
        # Plot Grad-CAM heatmap
        axes[i, 1].imshow(cam_resized, cmap='jet')
        axes[i, 1].set_title(f'Grad-CAM Heatmap', fontsize=10)
        axes[i, 1].axis('off')
        
        # Plot overlay
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title(f'Overlay\nPred: {CLASS_NAMES[pred_class]} ({pred_prob:.2%})', fontsize=10)
        axes[i, 2].axis('off')
    
    plt.suptitle(f'Grad-CAM Explainability: {CLASS_FULL_NAMES[target_class]} Samples', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Explainability samples saved to: {output_path}")


def run_evaluation(
    model_path: str = "models/best_model_resnet.pth",
    data_dir: str = "data/processed",
    output_dir: str = "reports",
    num_workers: int = 2
):
    """
    Run complete evaluation pipeline.
    
    Args:
        model_path: Path to saved model.
        data_dir: Directory containing test data.
        output_dir: Directory for outputs.
        num_workers: Number of data loading workers (use 0 for Prefect).
    """
    print("="*60)
    print("DERMAOPS MODEL EVALUATION")
    print("="*60)
    
    # Setup
    device = setup_device()
    
    # Load model
    model = load_model(model_path, device)
    
    # Load test data
    test_loader, test_dataset = get_test_dataloader(data_dir, num_workers=num_workers)
    
    # Run inference
    y_pred, y_true, y_probs = run_inference(model, test_loader, device)
    
    # Calculate and save metrics
    metrics = calculate_metrics(
        y_true, y_pred,
        output_path=f"{output_dir}/metrics_report.txt"
    )
    
    # Generate confusion matrix
    plot_confusion_matrix(
        y_true, y_pred,
        output_path=f"{output_dir}/figures/confusion_matrix_test.png"
    )
    
    # Generate Grad-CAM samples
    generate_gradcam_samples(
        model, test_dataset, device,
        output_path=f"{output_dir}/figures/explainability_samples.png",
        num_samples=3,
        target_class='mel'
    )
    
    # Final summary
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"\n📁 Generated Artifacts:")
    print(f"   📄 {output_dir}/metrics_report.txt")
    print(f"   📊 {output_dir}/figures/confusion_matrix_test.png")
    print(f"   🔬 {output_dir}/figures/explainability_samples.png")
    
    print(f"\n📊 Final Test Performance:")
    print(f"   [RESNET50-TEST] Accuracy: {metrics['accuracy']:.2f}, F1: {metrics['f1_weighted']:.2f}")
    
    return metrics


if __name__ == "__main__":
    # Get project root directory
    project_root = Path(__file__).parent.parent.parent
    os.chdir(project_root)
    
    print(f"Working directory: {os.getcwd()}")
    
    # Run evaluation
    metrics = run_evaluation()
    
    print("\n✅ Evaluation complete!")
