"""
Download HAM10000 Dataset from Kaggle.

This script downloads the HAM10000 skin cancer dataset and stores it in data/raw/.
"""

import os
import shutil
from pathlib import Path

import kagglehub


def download_ham10000(target_dir: str = "data/raw") -> str:
    """
    Download the HAM10000 dataset from Kaggle.
    
    Args:
        target_dir: Target directory to store the dataset.
        
    Returns:
        Path to the downloaded dataset.
    """
    # Create target directory if it doesn't exist
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    print("Downloading HAM10000 dataset from Kaggle...")
    print("This may take a while depending on your internet connection.")
    
    # Download the dataset using kagglehub
    dataset_path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
    
    print(f"Dataset downloaded to: {dataset_path}")
    
    # Copy files to target directory
    print(f"Copying files to {target_dir}...")
    
    # List downloaded files
    downloaded_files = list(Path(dataset_path).rglob("*"))
    print(f"Found {len(downloaded_files)} files/directories")
    
    for item in Path(dataset_path).iterdir():
        dest = target_path / item.name
        if item.is_file():
            shutil.copy2(item, dest)
            print(f"  Copied: {item.name}")
        elif item.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
            print(f"  Copied directory: {item.name}")
    
    print(f"\nDataset successfully saved to: {target_path.absolute()}")
    
    # List contents of target directory
    print("\nContents of data/raw/:")
    for item in sorted(target_path.iterdir()):
        if item.is_file():
            size_mb = item.stat().st_size / (1024 * 1024)
            print(f"  {item.name}: {size_mb:.2f} MB")
        else:
            num_files = len(list(item.rglob("*")))
            print(f"  {item.name}/ ({num_files} files)")
    
    return str(target_path)


if __name__ == "__main__":
    # Get project root directory
    project_root = Path(__file__).parent.parent.parent
    os.chdir(project_root)
    
    download_ham10000("data/raw")
