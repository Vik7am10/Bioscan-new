#!/usr/bin/env python3
"""
Find image using BIN ID instead of process ID.
"""

import os
import json
import pandas as pd
from pathlib import Path

def find_image_by_bin():
    print("🔍 SEARCHING FOR BLACK WITCH MOTH IMAGE BY BIN ID")
    print("=" * 60)
    
    bin_id = "BOLD:AAA5595"
    print(f"🏷️  Target BIN ID: {bin_id}")
    print(f"📊 Species: Ascalapha odorata (Black Witch Moth)")
    print(f"⚖️  Biomass: 647.975 mg")
    
    # Clean BIN ID for file searching (remove colon)
    clean_bin_id = bin_id.replace(":", "")
    alt_bin_id = bin_id.replace("BOLD:", "")
    
    print(f"\n🔍 Searching for files containing:")
    print(f"   - {bin_id}")
    print(f"   - {clean_bin_id}")
    print(f"   - {alt_bin_id}")
    print(f"   - AAA5595")
    
    # Search patterns
    search_terms = [
        bin_id,           # BOLD:AAA5595
        clean_bin_id,     # BOLDAAA5595
        alt_bin_id,       # AAA5595
        "AAA5595"         # Just the numeric part
    ]
    
    # Search in current directory and subdirectories
    print(f"\n📂 Searching in current directory and subdirectories...")
    
    found_files = []
    search_dirs = ['.', './images', './data', './bioscan_data']
    
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            print(f"   Searching in: {search_dir}")
            for root, dirs, files in os.walk(search_dir):
                for file in files:
                    file_lower = file.lower()
                    for term in search_terms:
                        if term.lower() in file_lower:
                            full_path = os.path.join(root, file)
                            found_files.append((term, full_path))
                            print(f"   ✅ Found: {full_path}")
    
    # Also check if there are any image directories we might have missed
    print(f"\n🗂️  Looking for common image directory names...")
    possible_dirs = [
        './bioscan_images', './BIOSCAN', './images', './photos', 
        './specimens', './data/images', './dataset/images'
    ]
    
    for dir_path in possible_dirs:
        if os.path.exists(dir_path):
            print(f"   Found directory: {dir_path}")
            # Quick search in this directory
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    file_lower = file.lower()
                    for term in search_terms:
                        if term.lower() in file_lower:
                            full_path = os.path.join(root, file)
                            found_files.append((term, full_path))
                            print(f"   ✅ Found: {full_path}")
                break  # Don't go too deep initially
    
    # Check image index more thoroughly
    print(f"\n📇 Checking image index for BIN-related entries...")
    try:
        with open('image_index.json', 'r') as f:
            image_index = json.load(f)
        
        # Look for any entries that might be related to this BIN
        bin_related = []
        for process_id, image_path in image_index.items():
            for term in search_terms:
                if term.lower() in image_path.lower() or term.lower() in process_id.lower():
                    bin_related.append((process_id, image_path))
                    print(f"   📸 Index match: {process_id} -> {image_path}")
        
        if not bin_related:
            print(f"   ❌ No BIN-related entries found in image index")
            
    except Exception as e:
        print(f"   ❌ Error checking image index: {e}")
    
    # Try to find any specimens from the same family/genus
    print(f"\n🔍 Looking for other Erebidae family specimens...")
    df = pd.read_csv('bin_results/bin_id_biomass_mapping_cleaned.csv')
    erebidae_specimens = df[df['most_common_family'] == 'Erebidae']
    
    print(f"   Found {len(erebidae_specimens)} Erebidae specimens in dataset")
    if len(erebidae_specimens) > 1:
        print(f"   Other large Erebidae specimens:")
        large_erebidae = erebidae_specimens[erebidae_specimens['mean_weight'] > 50].sort_values('mean_weight', ascending=False)
        for _, row in large_erebidae.head(5).iterrows():
            print(f"   - {row['dna_bin']}: {row['mean_weight']:.1f}mg ({row['most_common_species']})")
    
    if found_files:
        print(f"\n🎉 FOUND {len(found_files)} MATCHING FILES!")
        return found_files
    else:
        print(f"\n😞 No image files found using BIN ID search")
        print(f"\n💡 Suggestions:")
        print(f"   1. The image might be in a directory not yet searched")
        print(f"   2. The image filename might use a different naming convention")
        print(f"   3. Try searching online for 'Ascalapha odorata' images")
        print(f"   4. Check if there's a separate image dataset or archive")
        return []

if __name__ == "__main__":
    find_image_by_bin()