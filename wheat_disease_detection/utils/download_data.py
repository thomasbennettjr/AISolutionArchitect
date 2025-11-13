"""
Download sample wheat disease dataset from Kaggle
Requires Kaggle API credentials
"""

import os
import zipfile
import shutil


def setup_kaggle_credentials():
    """
    Setup Kaggle API credentials
    Instructions: 
    1. Go to https://www.kaggle.com/account
    2. Click "Create New API Token"
    3. Place kaggle.json in ~/.kaggle/ directory
    """
    kaggle_dir = os.path.expanduser('~/.kaggle')
    kaggle_json = os.path.join(kaggle_dir, 'kaggle.json')

    if not os.path.exists(kaggle_json):
        print("Kaggle credentials not found!")
        print("\nTo setup:")
        print("1. Visit https://www.kaggle.com/account")
        print("2. Click 'Create New API Token'")
        print("3. Place kaggle.json in ~/.kaggle/")
        return False

    # Set permissions
    os.chmod(kaggle_json, 0o600)
    print("✓ Kaggle credentials found")
    return True


def download_wheat_dataset():
    """
    Download wheat disease dataset
    Note: Update with actual dataset name when available
    """
    if not setup_kaggle_credentials():
        return

    print("\nDownloading wheat disease dataset...")
    print("Note: Replace with actual dataset name")

    # Example command (update with real dataset)
    # os.system('kaggle datasets download -d username/wheat-disease-dataset')

    print("\nManual alternative:")
    print("1. Visit Kaggle and search for wheat disease datasets")
    print("2. Download manually")
    print("3. Extract to data/wheat_diseases/ folder")


def organize_dataset(source_dir, output_dir):
    """
    Organize downloaded dataset into train/test structure

    Args:
        source_dir: Directory containing downloaded images
        output_dir: Output directory for organized data
    """
    print(f"\nOrganizing dataset from {source_dir} to {output_dir}")

    # Create directory structure
    for split in ['train', 'test']:
        for disease in ['Healthy', 'Leaf_Rust', 'Stem_Rust', 'Yellow_Rust', 'Septoria']:
            os.makedirs(os.path.join(
                output_dir, split, disease), exist_ok=True)

    print("✓ Directory structure created")
    print("\nPlease manually organize your images into the created folders")


if __name__ == "__main__":
    print("="*80)
    print("WHEAT DISEASE DATASET SETUP")
    print("="*80)

    # Check for Kaggle credentials
    setup_kaggle_credentials()

    # Create data directories
    data_dir = "data/wheat_diseases"
    organize_dataset("data/downloads", data_dir)

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n1. Download wheat disease images from:")
    print("   - Kaggle")
    print("   - PlantVillage")
    print("   - Agricultural research databases")
    print("   - Your own drone/aerial photos")
    print("\n2. Organize images into:")
    print(f"   {data_dir}/train/[Disease_Name]/")
    print(f"   {data_dir}/test/[Disease_Name]/")
    print("\n3. Run: python utils/data_utils.py")
    print("   To validate your dataset structure")
    print("\n4. Run: python train.py")
    print("   To start training")
