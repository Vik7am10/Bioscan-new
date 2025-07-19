#!/usr/bin/env python3
"""
BIOSCAN Dataset Download Script for Login Node
Downloads the full BIOSCAN-5M dataset with robust error handling
"""

import os
import sys
import time
from pathlib import Path

def download_split_with_retries(split, root_dir, max_retries=3):
    """Download a specific split with retry logic"""
    from bioscan_dataset import BIOSCAN5M
    
    for attempt in range(max_retries):
        print(f"🔄 Attempt {attempt + 1}/{max_retries} for {split} split...")
        
        try:
            dataset = BIOSCAN5M(
                root=root_dir,
                split=split,
                download=True,
                modality=('image',)
            )
            print(f"✅ Successfully downloaded {split} split!")
            print(f"Dataset size: {len(dataset):,} samples")
            return True
            
        except Exception as e:
            print(f"❌ Download failed for {split}: {e}")
            if attempt < max_retries - 1:
                print("⏳ Waiting 60 seconds before retry...")
                time.sleep(60)
            
    print(f"❌ Failed to download {split} split after {max_retries} attempts")
    return False

def check_disk_space(path):
    """Check available disk space"""
    statvfs = os.statvfs(path)
    free_bytes = statvfs.f_frsize * statvfs.f_bavail
    free_gb = free_bytes / (1024**3)
    
    print(f"💾 Available disk space: {free_gb:.1f} GB")
    return free_gb

def main():
    """Main download function"""
    root_dir = "./bioscan_data"
    
    print("=" * 50)
    print("📥 BIOSCAN-5M Dataset Download (Login Node)")
    print("=" * 50)
    
    # Test import
    try:
        from bioscan_dataset import BIOSCAN5M
        print("✅ bioscan-dataset package available")
    except ImportError as e:
        print(f"❌ Failed to import bioscan-dataset: {e}")
        print("Please install: pip install bioscan-dataset")
        return False
    
    # Check initial disk space
    free_space = check_disk_space(".")
    if free_space < 20:
        print(f"⚠️  Warning: Only {free_space:.1f}GB available. BIOSCAN-5M needs ~20GB")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            return False
    
    # Create directory structure
    os.makedirs(f"{root_dir}/bioscan5m/images/cropped_256", exist_ok=True)
    
    # Download each split
    splits = ['train', 'val', 'test']
    success_count = 0
    
    for split in splits:
        print(f"\n📥 Downloading {split} split...")
        
        if download_split_with_retries(split, root_dir):
            success_count += 1
            
            # Check disk usage after each split
            disk_usage = os.popen(f"du -sh {root_dir}").read().strip()
            print(f"💾 Disk usage after {split}: {disk_usage}")
            
            # Check remaining space
            remaining_space = check_disk_space(".")
            if remaining_space < 5:
                print(f"⚠️  Low disk space: {remaining_space:.1f}GB remaining")
        else:
            print(f"❌ Failed to download {split} split")
    
    # Verify complete dataset
    print("\n🔍 Verifying downloaded dataset...")
    
    total_samples = 0
    for split in splits:
        try:
            dataset = BIOSCAN5M(root=root_dir, split=split, download=False)
            samples = len(dataset)
            total_samples += samples
            print(f"✅ {split}: {samples:,} samples")
        except Exception as e:
            print(f"❌ {split}: Failed to verify - {e}")
    
    print(f"\n📊 Total dataset size: {total_samples:,} samples")
    
    # Show final directory structure
    print("\n📁 Final directory structure:")
    os.system(f"find {root_dir} -type d | head -15")
    
    # Count files
    print("\n📊 File counts:")
    os.system(f"find {root_dir} -name '*.jpg' | wc -l | xargs echo 'Images:'")
    os.system(f"find {root_dir} -name '*.csv' | wc -l | xargs echo 'CSV files:'")
    
    # Create download summary
    with open("bioscan_download_summary.txt", "w") as f:
        f.write("BIOSCAN-5M Download Summary\n")
        f.write("=" * 30 + "\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Successful splits: {success_count}/{len(splits)}\n")
        f.write(f"Total samples: {total_samples:,}\n")
        
        disk_usage = os.popen(f"du -sh {root_dir}").read().strip()
        f.write(f"Total disk usage: {disk_usage}\n")
    
    if success_count == len(splits):
        print("\n🎉 BIOSCAN dataset download completed successfully!")
        print("✅ Ready for BIN mapping and ML training")
        return True
    else:
        print(f"\n⚠️  Partial download: {success_count}/{len(splits)} splits successful")
        print("❌ Some splits failed - check error messages above")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)