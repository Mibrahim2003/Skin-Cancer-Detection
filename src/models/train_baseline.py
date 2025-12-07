"""
Baseline Model Training - Random Forest Classifier.

This script trains a simple Random Forest baseline model on flattened image data.
It serves as a comparison point for the deep learning models.
"""

import os
import time
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from tqdm import tqdm


# Class mapping for HAM10000
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def load_images_from_folder(
    data_dir: str,
    target_size: Tuple[int, int] = (64, 64)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load images from ImageNet-style folder structure and flatten them.
    
    Args:
        data_dir: Directory containing class subfolders.
        target_size: Size to resize images to before flattening.
        
    Returns:
        Tuple of (X, y) where X is flattened images and y is labels.
    """
    data_path = Path(data_dir)
    
    images = []
    labels = []
    
    print(f"Loading images from {data_dir}...")
    
    # Count total images for progress bar
    total_images = sum(
        len(list((data_path / class_name).glob("*.jpg")))
        for class_name in CLASS_NAMES
        if (data_path / class_name).exists()
    )
    
    with tqdm(total=total_images, desc="Loading") as pbar:
        for class_name in CLASS_NAMES:
            class_dir = data_path / class_name
            
            if not class_dir.exists():
                print(f"  Warning: {class_name} folder not found")
                continue
            
            class_idx = CLASS_TO_IDX[class_name]
            
            for img_path in class_dir.glob("*.jpg"):
                try:
                    # Load image
                    img = Image.open(img_path).convert('RGB')
                    
                    # Resize to smaller size for baseline
                    img = img.resize(target_size, Image.Resampling.LANCZOS)
                    
                    # Convert to numpy array and flatten
                    img_array = np.array(img).flatten()
                    
                    # Normalize to [0, 1]
                    img_array = img_array / 255.0
                    
                    images.append(img_array)
                    labels.append(class_idx)
                    
                except Exception as e:
                    print(f"  Error loading {img_path}: {e}")
                
                pbar.update(1)
    
    X = np.array(images)
    y = np.array(labels)
    
    print(f"  Loaded {len(X)} images with shape {X.shape}")
    
    return X, y


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    n_jobs: int = -1,
    random_state: int = 42
) -> RandomForestClassifier:
    """
    Train a Random Forest classifier.
    
    Args:
        X_train: Training features (flattened images).
        y_train: Training labels.
        n_estimators: Number of trees in the forest.
        n_jobs: Number of parallel jobs (-1 for all cores).
        random_state: Random seed for reproducibility.
        
    Returns:
        Trained RandomForestClassifier.
    """
    print(f"\nTraining Random Forest with {n_estimators} estimators...")
    
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=1,
        class_weight='balanced'  # Handle class imbalance
    )
    
    start_time = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    print(f"  Training completed in {train_time:.2f} seconds")
    
    return clf


def evaluate_model(
    clf: RandomForestClassifier,
    X_val: np.ndarray,
    y_val: np.ndarray
) -> dict:
    """
    Evaluate the model on validation data.
    
    Args:
        clf: Trained classifier.
        X_val: Validation features.
        y_val: Validation labels.
        
    Returns:
        Dictionary with evaluation metrics.
    """
    print("\nEvaluating on validation set...")
    
    # Predictions
    y_pred = clf.predict(X_val)
    
    # Calculate metrics
    accuracy = accuracy_score(y_val, y_pred)
    f1_weighted = f1_score(y_val, y_pred, average='weighted')
    f1_macro = f1_score(y_val, y_pred, average='macro')
    
    # Detailed report
    report = classification_report(
        y_val, y_pred,
        target_names=CLASS_NAMES,
        digits=4
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'classification_report': report,
        'confusion_matrix': cm
    }
    
    return metrics


def print_results(metrics: dict):
    """Print formatted evaluation results."""
    print("\n" + "="*60)
    print("[BASELINE] Random Forest Results")
    print("="*60)
    
    print(f"\n📊 Overall Metrics:")
    print(f"   Accuracy:    {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"   F1 Weighted: {metrics['f1_weighted']:.4f}")
    print(f"   F1 Macro:    {metrics['f1_macro']:.4f}")
    
    print(f"\n📋 Classification Report:")
    print(metrics['classification_report'])
    
    print(f"\n🔢 Confusion Matrix:")
    print(f"   Classes: {CLASS_NAMES}")
    print(metrics['confusion_matrix'])
    
    # Summary line for easy comparison
    print("\n" + "="*60)
    print(f"[BASELINE] Random Forest - Val Acc: {metrics['accuracy']:.2f}, "
          f"F1: {metrics['f1_weighted']:.2f}")
    print("="*60)


def run_baseline_training(
    data_dir: str = "data/processed",
    image_size: int = 64,
    n_estimators: int = 100
):
    """
    Run the complete baseline training pipeline.
    
    Args:
        data_dir: Directory containing processed train/val data.
        image_size: Size to resize images to (will be squared).
        n_estimators: Number of trees for Random Forest.
    """
    print("="*60)
    print("BASELINE MODEL TRAINING - Random Forest")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Image size: {image_size}x{image_size}")
    print(f"  Feature vector size: {image_size * image_size * 3}")
    print(f"  Random Forest estimators: {n_estimators}")
    
    # Load training data
    train_dir = Path(data_dir) / "train"
    X_train, y_train = load_images_from_folder(
        str(train_dir),
        target_size=(image_size, image_size)
    )
    
    # Load validation data
    val_dir = Path(data_dir) / "val"
    X_val, y_val = load_images_from_folder(
        str(val_dir),
        target_size=(image_size, image_size)
    )
    
    # Print class distribution
    print(f"\nClass Distribution:")
    print(f"  {'Class':<8} {'Train':>8} {'Val':>8}")
    print(f"  {'-'*26}")
    for idx, name in enumerate(CLASS_NAMES):
        train_count = (y_train == idx).sum()
        val_count = (y_val == idx).sum()
        print(f"  {name:<8} {train_count:>8} {val_count:>8}")
    
    # Train model
    clf = train_random_forest(X_train, y_train, n_estimators=n_estimators)
    
    # Evaluate
    metrics = evaluate_model(clf, X_val, y_val)
    
    # Print results
    print_results(metrics)
    
    return clf, metrics


if __name__ == "__main__":
    # Get project root directory
    project_root = Path(__file__).parent.parent.parent
    os.chdir(project_root)
    
    print(f"Working directory: {os.getcwd()}")
    
    # Run baseline training
    clf, metrics = run_baseline_training()
    
    print("\n✅ Baseline training complete!")
