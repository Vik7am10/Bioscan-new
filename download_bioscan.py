#!/usr/bin/env python3
"""
Download BIOSCAN dataset to scratch directory
"""
import os
import sys
from datetime import datetime
from bioscan_dataset import BIOSCAN5M

def download_bioscan():
    print(f"📥 BIOSCAN Download Started: {datetime.now()}")
    
    # Navigate to scratch directory
    scratch_dir = os.path.expanduser("~/links/scratch")
    os.chdir(scratch_dir)
    
    # Create bioscan_data directory
    data_dir = os.path.join(scratch_dir, "bioscan_data")
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"📁 Download directory: {data_dir}")
    
    splits = ['train', 'val', 'test']
    
    for split in splits:
        try:
            print(f"\n🔄 Downloading {split} split...")
            start_time = datetime.now()
            
            dataset = BIOSCAN5M(
                root=data_dir,
                split=split,
                download=True,
                modality=('image',)
            )
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            print(f"✅ {split} split completed: {len(dataset):,} samples in {duration}")
            
        except Exception as e:
            print(f"❌ {split} split failed: {e}")
            continue
    
    print(f"\n🎯 Download completed: {datetime.now()}")

if __name__ == "__main__":
    download_bioscan()