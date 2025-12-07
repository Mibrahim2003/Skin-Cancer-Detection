"""
Data Ingestion Pipeline for HAM10000 Dataset.

This script handles downloading and validating the HAM10000 skin cancer dataset.
It authenticates with Kaggle API and downloads the dataset to data/raw/.
"""

import os
import shutil
from pathlib import Path

import kagglehub


def verify_dataset(data_dir: str = "data/raw") -> bool:
    """
    Verify that the dataset was downloaded correctly.
    
    Args:
        data_dir: Directory containing the dataset.
        
    Returns:
        True if verification passes, False otherwise.
    """
    data_path = Path(data_dir)
    
    # Required files and directories
    required_files = [
        "HAM10000_metadata.csv",
    ]
    
    required_dirs = [
        "HAM10000_images_part_1",
        "HAM10000_images_part_2",
    ]
    
    print("\n" + "="*50)
    print("VERIFICATION RESULTS")
    print("="*50)
    
    all_present = True
    
    # Check required files
    for filename in required_files:
        filepath = data_path / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"✅ {filename} ({size_mb:.2f} MB)")
        else:
            print(f"❌ {filename} - MISSING")
            all_present = False
    
    # Check required directories
    for dirname in required_dirs:
        dirpath = data_path / dirname
        if dirpath.exists() and dirpath.is_dir():
            num_files = len(list(dirpath.glob("*.jpg")))
            print(f"✅ {dirname}/ ({num_files} images)")
        else:
            print(f"❌ {dirname}/ - MISSING")
            all_present = False
    
    print("="*50)
    
    if all_present:
        print("✅ Dataset verification PASSED!")
    else:
        print("❌ Dataset verification FAILED!")
    
    return all_present


def ingest_ham10000(target_dir: str = "data/raw") -> str:
    """
    Download and ingest the HAM10000 dataset from Kaggle.
    
    This function:
    1. Authenticates using Kaggle API (assumes kaggle.json is present or
       KAGGLE_USERNAME/KAGGLE_KEY environment variables are set)
    2. Downloads the "kmader/skin-cancer-mnist-ham10000" dataset
    3. Unzips it into data/raw/
    4. Verifies the download by checking for HAM10000_metadata.csv
    
    Args:
        target_dir: Target directory to store the dataset.
        
    Returns:
        Path to the downloaded dataset.
        
    Raises:
        FileNotFoundError: If verification fails after download.
    """
    # Create target directory if it doesn't exist
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Check if dataset already exists
    metadata_file = target_path / "HAM10000_metadata.csv"
    if metadata_file.exists():
        print("Dataset already exists. Verifying...")
        if verify_dataset(target_dir):
            return str(target_path)
        print("\nRe-downloading dataset...")
    
    print("="*50)
    print("HAM10000 DATA INGESTION PIPELINE")
    print("="*50)
    print("\nDownloading HAM10000 dataset from Kaggle...")
    print("Dataset: kmader/skin-cancer-mnist-ham10000")
    print("This may take a while depending on your internet connection.\n")
    
    # Download the dataset using kagglehub
    # kagglehub handles authentication via:
    # - ~/.kaggle/kaggle.json file
    # - KAGGLE_USERNAME and KAGGLE_KEY environment variables
    dataset_path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
    
    print(f"\nDataset downloaded to cache: {dataset_path}")
    
    # Copy files to target directory
    print(f"\nCopying files to {target_dir}...")
    
    for item in Path(dataset_path).iterdir():
        dest = target_path / item.name
        if item.is_file():
            shutil.copy2(item, dest)
            print(f"  📄 Copied: {item.name}")
        elif item.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
            num_files = len(list(item.glob("*")))
            print(f"  📁 Copied: {item.name}/ ({num_files} files)")
    
    # Verify the download
    if not verify_dataset(target_dir):
        raise FileNotFoundError(
            "Dataset verification failed! Required files are missing. "
            "Please check your Kaggle authentication and try again."
        )
    
    return str(target_path)


if __name__ == "__main__":
    # Get project root directory
    project_root = Path(__file__).parent.parent.parent
    os.chdir(project_root)
    
    print(f"Working directory: {os.getcwd()}")
    
    # Run ingestion
    result_path = ingest_ham10000("data/raw")
    print(f"\n✅ Data ingestion complete! Dataset saved to: {result_path}")
