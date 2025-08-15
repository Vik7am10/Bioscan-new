#!/usr/bin/env python3
"""
Find the largest biomass specimen and locate its image file.
"""

import pandas as pd
import json
import os
from pathlib import Path

def find_largest_specimen():
    print("🔍 FINDING LARGEST BIOMASS SPECIMEN")
    print("=" * 50)
    
    # Load cleaned dataset
    df = pd.read_csv('bin_results/bin_id_biomass_mapping_cleaned.csv')
    
    # Find the row with maximum biomass
    max_idx = df['mean_weight'].idxmax()
    largest_specimen = df.loc[max_idx]
    
    print(f"📊 Largest specimen details:")
    print(f"   BIN ID: {largest_specimen['dna_bin']}")
    print(f"   Biomass: {largest_specimen['mean_weight']:.3f} mg")
    print(f"   Specimen count: {largest_specimen['specimen_count']}")
    print(f"   Species: {largest_specimen['most_common_species']}")
    print(f"   Genus: {largest_specimen['most_common_genus']}")
    print(f"   Family: {largest_specimen['most_common_family']}")
    print(f"   Order: {largest_specimen['most_common_order']}")
    
    # Get the BIOSCAN process IDs for this BIN
    bioscan_ids = eval(largest_specimen['bioscan_processids']) if isinstance(largest_specimen['bioscan_processids'], str) else largest_specimen['bioscan_processids']
    print(f"\n📋 BIOSCAN Process IDs ({len(bioscan_ids)} specimens):")
    for i, pid in enumerate(bioscan_ids[:5]):  # Show first 5
        print(f"   {i+1}. {pid}")
    if len(bioscan_ids) > 5:
        print(f"   ... and {len(bioscan_ids) - 5} more")
    
    # Load image index to find image paths
    print(f"\n🖼️  Looking for image files...")
    
    try:
        with open('image_index.json', 'r') as f:
            image_index = json.load(f)
        print(f"   Loaded image index with {len(image_index)} entries")
        
        # Find images for these process IDs
        found_images = []
        for pid in bioscan_ids:
            if pid in image_index:
                image_path = image_index[pid]
                found_images.append((pid, image_path))
                
        print(f"\n📸 Found {len(found_images)} image files:")
        
        if found_images:
            # Take the first available image
            first_pid, first_image = found_images[0]
            print(f"   Process ID: {first_pid}")
            print(f"   Image path: {first_image}")
            
            # Check if image file exists
            if os.path.exists(first_image):
                print(f"   ✅ Image file exists!")
                
                # Create a symlink or copy for easy access
                output_name = f"largest_specimen_{largest_specimen['mean_weight']:.1f}mg_{first_pid}.jpg"
                
                # Try to create a symlink first, fallback to copy
                try:
                    if os.path.exists(output_name):
                        os.remove(output_name)
                    os.symlink(first_image, output_name)
                    print(f"   🔗 Created symlink: {output_name}")
                except:
                    # Fallback to copying
                    import shutil
                    shutil.copy2(first_image, output_name)
                    print(f"   📁 Copied image to: {output_name}")
                
                return output_name, largest_specimen
                
            else:
                print(f"   ❌ Image file not found at: {first_image}")
                
        else:
            print(f"   ❌ No image files found for these process IDs")
            
    except FileNotFoundError:
        print(f"   ❌ Image index file not found")
    except Exception as e:
        print(f"   ❌ Error loading image index: {e}")
    
    return None, largest_specimen

if __name__ == "__main__":
    image_file, specimen_info = find_largest_specimen()
    
    if image_file:
        print(f"\n🎉 SUCCESS!")
        print(f"   The largest specimen image is now available as: {image_file}")
        print(f"   You can download it with: scp rorqual:~/path/to/{image_file} ./")
    else:
        print(f"\n😞 Could not locate the image file")
        print(f"   BIN ID: {specimen_info['dna_bin']}")
        print(f"   Try searching for images manually using the BIOSCAN process IDs")