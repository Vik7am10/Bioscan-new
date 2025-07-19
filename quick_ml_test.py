#!/usr/bin/env python3
"""
Quick ML test with limited samples to verify the pipeline works.
"""

import os
import sys
import pandas as pd
from pathlib import Path
import subprocess

def main():
    print("🚀 Quick ML Test - Limited Samples")
    print("=" * 50)
    
    # Check if BIN mapping exists
    bin_mapping_path = "bin_results/bin_id_biomass_mapping.csv"
    if not os.path.exists(bin_mapping_path):
        print(f"❌ BIN mapping file not found: {bin_mapping_path}")
        return
    
    # Check if images exist
    images_root = "/home/vikramre/scratch/bioscan_data/bioscan5m/images"
    if not os.path.exists(images_root):
        print(f"❌ Images directory not found: {images_root}")
        return
    
    print(f"✅ Found BIN mapping: {bin_mapping_path}")
    print(f"✅ Found images directory: {images_root}")
    
    # Run with limited samples
    cmd = [
        "python", "direct_ml_pipeline.py",
        "--bin_mapping_csv", bin_mapping_path,
        "--bioscan_images_root", images_root,
        "--model_type", "basic",  # Use basic model for speed
        "--batch_size", "8",
        "--num_epochs", "2",
        "--max_samples", "50",  # Very limited for testing
        "--output_dir", "quick_test_output"
    ]
    
    print(f"🔄 Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 min timeout
        print("✅ Command completed successfully!")
        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
    except subprocess.TimeoutExpired:
        print("❌ Command timed out after 5 minutes - likely hanging")
    except Exception as e:
        print(f"❌ Error running command: {e}")

if __name__ == "__main__":
    main()