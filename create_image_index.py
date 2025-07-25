#!/usr/bin/env python3
"""
Create image index to avoid slow glob operations.
Pre-scan image directories once and save mappings.
"""

import os
import json
import time
from pathlib import Path

def create_image_index(images_root: str, output_file: str = "image_index.json"):
    """
    Create index of all image files in BIOSCAN directory.
    Maps processid -> image_path for fast lookups.
    """
    print(f"🔍 Scanning images in: {images_root}")
    
    image_index = {}
    total_files = 0
    
    # Search in cropped_256 directory
    cropped_dir = Path(images_root) / "cropped_256"
    
    if not cropped_dir.exists():
        print(f"❌ Directory not found: {cropped_dir}")
        return
    
    print(f"📁 Scanning directory: {cropped_dir}")
    
    start_time = time.time()
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(cropped_dir):
        for file in files:
            if file.endswith('.jpg'):
                # Extract processid from filename (without extension)
                processid = file.replace('.jpg', '')
                full_path = os.path.join(root, file)
                
                # Store mapping
                image_index[processid] = full_path
                total_files += 1
                
                if total_files % 10000 == 0:
                    print(f"  Processed {total_files} images...")
    
    end_time = time.time()
    
    print(f"✅ Indexing complete!")
    print(f"   - Total images: {total_files}")
    print(f"   - Time taken: {end_time - start_time:.2f} seconds")
    
    # Save index to file
    with open(output_file, 'w') as f:
        json.dump(image_index, f, indent=2)
    
    print(f"💾 Index saved to: {output_file}")
    return image_index

def main():
    images_root = "/home/vikramre/links/scratch/bioscan_data/bioscan5m/images"
    
    if not os.path.exists(images_root):
        print(f"❌ Images directory not found: {images_root}")
        return
    
    print("🚀 Creating BIOSCAN Image Index")
    print("=" * 50)
    
    index = create_image_index(images_root)
    
    # Show some sample mappings
    print("\n📋 Sample mappings:")
    sample_keys = list(index.keys())[:5]
    for key in sample_keys:
        print(f"  {key} -> {index[key]}")

if __name__ == "__main__":
    main()