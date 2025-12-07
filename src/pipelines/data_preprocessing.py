"""
Data Preprocessing Pipeline for HAM10000 Dataset.

This script handles:
1. Loading metadata and mapping image paths
2. Stratified train/val/test splitting (80/10/10)
3. Image resizing to 224x224 for ResNet
4. Organizing data into ImageNet-style folder structure
"""

import os
import shutil
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def load_and_prepare_metadata(data_dir: str = "data/raw") -> pd.DataFrame:
    """
    Load metadata and create full image paths.
    
    Args:
        data_dir: Directory containing the raw dataset.
        
    Returns:
        DataFrame with image paths mapped.
    """
    data_path = Path(data_dir)
    metadata_path = data_path / "HAM10000_metadata.csv"
    
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found at {metadata_path}. "
            "Please run data_ingestion.py first."
        )
    
    df = pd.read_csv(metadata_path)
    print(f"✅ Loaded metadata: {len(df)} records")
    
    # Map image_id to actual file paths
    # Images are in HAM10000_images_part_1 and HAM10000_images_part_2
    def find_image_path(image_id: str) -> str:
        """Find the full path for an image ID."""
        for part in ["HAM10000_images_part_1", "HAM10000_images_part_2"]:
            img_path = data_path / part / f"{image_id}.jpg"
            if img_path.exists():
                return str(img_path)
        return None
    
    print("Mapping image paths...")
    df['path'] = df['image_id'].apply(find_image_path)
    
    # Check for missing images
    missing = df['path'].isnull().sum()
    if missing > 0:
        print(f"⚠️  Warning: {missing} images not found!")
        df = df.dropna(subset=['path'])
    
    print(f"✅ Successfully mapped {len(df)} images")
    return df


def stratified_split(
    df: pd.DataFrame,
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform stratified train/val/test split.
    
    Stratification ensures each split maintains the same class proportions
    as the original dataset - critical for imbalanced medical data.
    
    Args:
        df: Input DataFrame with 'dx' column for stratification.
        train_size: Proportion for training set.
        val_size: Proportion for validation set.
        test_size: Proportion for test set.
        random_state: Random seed for reproducibility.
        
    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, \
        "Split proportions must sum to 1.0"
    
    print("\n" + "="*50)
    print("STRATIFIED DATA SPLITTING")
    print("="*50)
    print(f"Split ratios: Train={train_size:.0%}, Val={val_size:.0%}, Test={test_size:.0%}")
    
    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df['dx'],
        random_state=random_state
    )
    
    # Second split: separate train and validation from remaining
    # Adjust val_size to account for the already removed test set
    adjusted_val_size = val_size / (train_size + val_size)
    
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=adjusted_val_size,
        stratify=train_val_df['dx'],
        random_state=random_state
    )
    
    # Verify stratification
    print("\nSplit Results:")
    print(f"  Train: {len(train_df):,} images ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(val_df):,} images ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test_df):,} images ({len(test_df)/len(df)*100:.1f}%)")
    
    # Verify class distribution is maintained
    print("\nClass Distribution Verification:")
    print("-" * 60)
    print(f"{'Class':<8} {'Original':>10} {'Train':>10} {'Val':>10} {'Test':>10}")
    print("-" * 60)
    
    for dx in sorted(df['dx'].unique()):
        orig_pct = (df['dx'] == dx).sum() / len(df) * 100
        train_pct = (train_df['dx'] == dx).sum() / len(train_df) * 100
        val_pct = (val_df['dx'] == dx).sum() / len(val_df) * 100
        test_pct = (test_df['dx'] == dx).sum() / len(test_df) * 100
        print(f"{dx:<8} {orig_pct:>9.1f}% {train_pct:>9.1f}% {val_pct:>9.1f}% {test_pct:>9.1f}%")
    
    print("-" * 60)
    print("✅ Stratification verified - class proportions maintained!")
    
    return train_df, val_df, test_df


def process_and_save_images(
    df: pd.DataFrame,
    split_name: str,
    output_dir: str = "data/processed",
    target_size: Tuple[int, int] = (224, 224)
) -> int:
    """
    Process images: resize and save in ImageNet folder structure.
    
    Args:
        df: DataFrame with 'path' and 'dx' columns.
        split_name: Name of split ('train', 'val', 'test').
        output_dir: Base output directory.
        target_size: Target image size (width, height).
        
    Returns:
        Number of images processed.
    """
    output_path = Path(output_dir) / split_name
    
    # Create class subdirectories
    for dx in df['dx'].unique():
        (output_path / dx).mkdir(parents=True, exist_ok=True)
    
    processed_count = 0
    errors = []
    
    print(f"\nProcessing {split_name} images...")
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  {split_name}"):
        try:
            # Load image
            img = cv2.imread(row['path'])
            
            if img is None:
                errors.append(row['image_id'])
                continue
            
            # Resize to target size (224x224 for ResNet)
            img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            
            # Save to class folder
            output_file = output_path / row['dx'] / f"{row['image_id']}.jpg"
            cv2.imwrite(str(output_file), img_resized)
            
            processed_count += 1
            
        except Exception as e:
            errors.append(f"{row['image_id']}: {str(e)}")
    
    if errors:
        print(f"  ⚠️  {len(errors)} errors encountered")
    
    return processed_count


def create_dataset_summary(output_dir: str = "data/processed"):
    """
    Create a summary of the processed dataset.
    
    Args:
        output_dir: Directory containing processed data.
    """
    output_path = Path(output_dir)
    
    print("\n" + "="*50)
    print("PROCESSED DATASET SUMMARY")
    print("="*50)
    
    total_images = 0
    
    for split in ['train', 'val', 'test']:
        split_path = output_path / split
        if not split_path.exists():
            continue
            
        print(f"\n{split.upper()}:")
        split_total = 0
        
        for class_dir in sorted(split_path.iterdir()):
            if class_dir.is_dir():
                count = len(list(class_dir.glob("*.jpg")))
                print(f"  {class_dir.name}: {count:,} images")
                split_total += count
        
        print(f"  {'─'*20}")
        print(f"  Total: {split_total:,} images")
        total_images += split_total
    
    print(f"\n{'='*50}")
    print(f"GRAND TOTAL: {total_images:,} images")
    print(f"Image size: 224x224 pixels (RGB)")
    print(f"Format: JPEG")


def run_preprocessing(
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    target_size: Tuple[int, int] = (224, 224),
    clear_existing: bool = True
):
    """
    Run the complete preprocessing pipeline.
    
    Args:
        raw_dir: Directory containing raw dataset.
        processed_dir: Directory for processed output.
        target_size: Target image size for resizing.
        clear_existing: Whether to clear existing processed data.
    """
    print("="*60)
    print("HAM10000 DATA PREPROCESSING PIPELINE")
    print("="*60)
    print(f"\nTarget image size: {target_size[0]}x{target_size[1]} pixels")
    
    # Clear existing processed data if requested
    processed_path = Path(processed_dir)
    if clear_existing and processed_path.exists():
        print(f"\nClearing existing processed data...")
        for split in ['train', 'val', 'test']:
            split_path = processed_path / split
            if split_path.exists():
                shutil.rmtree(split_path)
    
    # Step 1: Load and prepare metadata
    df = load_and_prepare_metadata(raw_dir)
    
    # Step 2: Stratified split
    train_df, val_df, test_df = stratified_split(df)
    
    # Step 3: Process and save images
    print("\n" + "="*50)
    print("IMAGE PROCESSING")
    print("="*50)
    
    train_count = process_and_save_images(train_df, "train", processed_dir, target_size)
    val_count = process_and_save_images(val_df, "val", processed_dir, target_size)
    test_count = process_and_save_images(test_df, "test", processed_dir, target_size)
    
    print(f"\n✅ Processed {train_count + val_count + test_count:,} images total")
    
    # Step 4: Create summary
    create_dataset_summary(processed_dir)
    
    # Save split information for reproducibility
    split_info = {
        'train_ids': train_df['image_id'].tolist(),
        'val_ids': val_df['image_id'].tolist(),
        'test_ids': test_df['image_id'].tolist()
    }
    
    split_file = processed_path / "split_info.csv"
    
    # Create a DataFrame with split assignments
    all_splits = []
    for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        temp = split_df[['image_id', 'dx']].copy()
        temp['split'] = split_name
        all_splits.append(temp)
    
    split_df = pd.concat(all_splits, ignore_index=True)
    split_df.to_csv(split_file, index=False)
    print(f"\n✅ Split information saved to: {split_file}")
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    # Get project root directory
    project_root = Path(__file__).parent.parent.parent
    os.chdir(project_root)
    
    print(f"Working directory: {os.getcwd()}")
    
    # Run preprocessing pipeline
    train_df, val_df, test_df = run_preprocessing()
    
    print("\n" + "="*60)
    print("✅ PREPROCESSING COMPLETE!")
    print("="*60)
    print("\nData is ready for model training.")
    print("Structure: data/processed/{train,val,test}/{class_name}/*.jpg")
