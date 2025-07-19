#!/usr/bin/env python3
"""
Fix BIOSCAN Dataset Download Script
This script attempts to properly download the BIOSCAN-5M metadata
"""

import os
import sys
import pandas as pd
from pathlib import Path

def test_bioscan_import():
    """Test if bioscan-dataset can be imported"""
    print("🔍 Testing bioscan-dataset import...")
    try:
        from bioscan_dataset import BIOSCAN5M
        print("✅ BIOSCAN5M import successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"⚠️ Import error: {e}")
        return False

def clean_empty_files(bioscan_root):
    """Remove empty metadata files"""
    print("🧹 Cleaning empty metadata files...")
    metadata_file = os.path.join(bioscan_root, "bioscan5m", "metadata", "csv", "BIOSCAN_5M_Insect_Dataset_metadata.csv")
    
    if os.path.exists(metadata_file):
        file_size = os.path.getsize(metadata_file)
        print(f"Current metadata file size: {file_size} bytes")
        
        if file_size == 0:
            print("🗑️ Removing empty metadata file")
            os.remove(metadata_file)
            return True
        else:
            print("✅ Metadata file has content")
            return False
    else:
        print("ℹ️ No metadata file found")
        return True

def attempt_download_method_1(bioscan_root):
    """Method 1: Full dataset download"""
    print("\n📥 Method 1: Attempting full dataset download...")
    try:
        from bioscan_dataset import BIOSCAN5M
        dataset = BIOSCAN5M(root=bioscan_root, download=True, split='train')
        print("✅ Full dataset download successful")
        print(f"Metadata shape: {dataset.metadata.shape}")
        print(f"Metadata columns: {list(dataset.metadata.columns)[:10]}...")
        return True
    except Exception as e:
        print(f"❌ Method 1 failed: {e}")
        return False

def attempt_download_method_2(bioscan_root):
    """Method 2: Minimal download (metadata only)"""
    print("\n📥 Method 2: Attempting minimal download...")
    try:
        from bioscan_dataset import BIOSCAN5M
        dataset = BIOSCAN5M(root=bioscan_root, download=True, modality=())
        print("✅ Minimal download successful")
        print(f"Metadata shape: {dataset.metadata.shape}")
        print(f"Metadata columns: {list(dataset.metadata.columns)[:10]}...")
        return True
    except Exception as e:
        print(f"❌ Method 2 failed: {e}")
        return False

def attempt_download_method_3(bioscan_root):
    """Method 3: Try different split"""
    print("\n📥 Method 3: Attempting with different split...")
    try:
        from bioscan_dataset import BIOSCAN5M
        dataset = BIOSCAN5M(root=bioscan_root, download=True, split='test')
        print("✅ Split test download successful")
        print(f"Metadata shape: {dataset.metadata.shape}")
        print(f"Metadata columns: {list(dataset.metadata.columns)[:10]}...")
        return True
    except Exception as e:
        print(f"❌ Method 3 failed: {e}")
        return False

def verify_metadata_file(bioscan_root):
    """Verify the metadata file is valid"""
    print("\n✅ Verifying metadata file...")
    metadata_file = os.path.join(bioscan_root, "bioscan5m", "metadata", "csv", "BIOSCAN_5M_Insect_Dataset_metadata.csv")
    
    if not os.path.exists(metadata_file):
        print("❌ Metadata file does not exist")
        return False
    
    file_size = os.path.getsize(metadata_file)
    print(f"📊 Metadata file size: {file_size:,} bytes")
    
    if file_size == 0:
        print("❌ Metadata file is empty")
        return False
    
    # Try to load it with pandas
    try:
        df = pd.read_csv(metadata_file)
        print(f"✅ Metadata loaded successfully: {len(df):,} records")
        print(f"Columns: {list(df.columns)[:10]}...")
        
        # Check for BIN IDs
        if 'dna_bin' in df.columns:
            bin_count = df['dna_bin'].notna().sum()
            print(f"🧬 Found {bin_count:,} records with BIN IDs")
        
        return True
    except Exception as e:
        print(f"❌ Error loading metadata: {e}")
        return False

def main():
    bioscan_root = "./bioscan_data"
    
    print("🔧 BIOSCAN Dataset Download Fix Script")
    print("=" * 50)
    
    # Test import
    if not test_bioscan_import():
        print("❌ Cannot proceed without bioscan-dataset package")
        sys.exit(1)
    
    # Clean empty files
    clean_empty_files(bioscan_root)
    
    # Try different download methods
    methods = [
        attempt_download_method_1,
        attempt_download_method_2,
        attempt_download_method_3
    ]
    
    success = False
    for method in methods:
        if method(bioscan_root):
            success = True
            break
    
    if not success:
        print("\n❌ All download methods failed")
        print("💡 Possible issues:")
        print("   - Network connectivity on compute node")
        print("   - BIOSCAN dataset server issues")
        print("   - Package version compatibility")
        sys.exit(1)
    
    # Verify the result
    if verify_metadata_file(bioscan_root):
        print("\n🎉 SUCCESS: BIOSCAN metadata downloaded and verified!")
        print("✅ Ready to run BIN mapping analysis")
    else:
        print("\n❌ FAILED: Metadata file is not valid")
        sys.exit(1)

if __name__ == "__main__":
    main()