#!/usr/bin/env python3
"""
Create cleaned dataset by removing biological outliers identified by scaling law analysis.
This script filters out samples that violate the sqrt(area) vs cube_root(mass) relationship.
"""

import pandas as pd
import json
import os

def create_cleaned_dataset():
    """Remove outlier samples and create cleaned BIN mapping."""
    
    print("🧹 Creating Cleaned Dataset")
    print("=" * 50)
    
    # Load outliers
    print("📁 Loading outlier data...")
    outliers_df = pd.read_csv('biomass_outliers.csv')
    print(f"   Found {len(outliers_df)} biological outliers")
    print(f"   Outlier columns: {outliers_df.columns.tolist()}")
    
    # Show sample outliers
    if len(outliers_df) > 0:
        print(f"\n📊 Sample outliers:")
        sample_outliers = outliers_df.head(3)
        for _, row in sample_outliers.iterrows():
            print(f"   {row['processid']}: {row['weight']:.1f}mg (predicted: {row['predicted_mass']:.1f}mg)")
    
    # Get outlier processids to exclude
    outlier_processids = set(outliers_df['processid'].tolist())
    print(f"\n🚫 Processids to exclude: {len(outlier_processids)}")
    
    # Load original BIN mapping
    print(f"\n📁 Loading original BIN mapping...")
    bin_df = pd.read_csv('bin_results/bin_id_biomass_mapping.csv')
    print(f"   Original dataset: {len(bin_df)} BIN groups")
    
    # Calculate total original samples
    original_total = 0
    for _, row in bin_df.iterrows():
        try:
            bioscan_processids = eval(row['bioscan_processids']) if isinstance(row['bioscan_processids'], str) else row['bioscan_processids']
            original_total += len(bioscan_processids)
        except:
            continue
    
    print(f"   Original total samples: {original_total}")
    
    # Create cleaned dataset by filtering out outliers
    print(f"\n🔄 Filtering outliers...")
    cleaned_bins = []
    removed_bins = 0
    clean_total = 0
    
    for _, row in bin_df.iterrows():
        try:
            # Get processids for this BIN
            if isinstance(row['bioscan_processids'], str):
                bioscan_processids = eval(row['bioscan_processids'])
            else:
                bioscan_processids = row['bioscan_processids']
            
            # Filter out outlier processids
            clean_processids = [pid for pid in bioscan_processids if pid not in outlier_processids]
            
            if clean_processids:  # Keep BIN if it has any clean samples
                row_clean = row.copy()
                row_clean['bioscan_processids'] = clean_processids
                row_clean['specimen_count'] = len(clean_processids)
                cleaned_bins.append(row_clean)
                clean_total += len(clean_processids)
            else:
                removed_bins += 1
                
        except Exception as e:
            print(f"   Warning: Error processing BIN {row.get('dna_bin', 'unknown')}: {e}")
            continue
    
    # Create cleaned DataFrame
    cleaned_df = pd.DataFrame(cleaned_bins)
    
    # Save cleaned dataset
    output_file = 'bin_results/bin_id_biomass_mapping_cleaned.csv'
    cleaned_df.to_csv(output_file, index=False)
    
    # Print summary
    print(f"\n✅ Cleaning Results:")
    print(f"   Original BIN groups: {len(bin_df)}")
    print(f"   Cleaned BIN groups: {len(cleaned_df)}")
    print(f"   Removed BIN groups: {removed_bins}")
    print(f"   Original samples: {original_total:,}")
    print(f"   Clean samples: {clean_total:,}")
    print(f"   Removed samples: {original_total - clean_total:,} ({(original_total - clean_total)/original_total*100:.1f}%)")
    
    print(f"\n💾 Saved cleaned dataset: {output_file}")
    
    # Verify the cleaning worked
    print(f"\n🔍 Verification:")
    print(f"   Expected outliers removed: {len(outlier_processids)}")
    print(f"   Actual samples removed: {original_total - clean_total}")
    
    if len(outlier_processids) == (original_total - clean_total):
        print(f"   ✅ Perfect match - all outliers removed")
    else:
        print(f"   ⚠️  Mismatch - some outliers may belong to same BIN groups")
    
    # Update baseline calculation
    print(f"\n📈 Updated Performance Target:")
    weights = []
    for _, row in cleaned_df.iterrows():
        weight = row['mean_weight']
        count = row['specimen_count'] 
        weights.extend([weight] * count)
    
    if weights:
        import numpy as np
        median_weight = np.median(weights)
        mean_weight = np.mean(weights)
        mae_baseline = np.mean(np.abs(np.array(weights) - median_weight))
        
        print(f"   Clean dataset median: {median_weight:.2f} mg")
        print(f"   Clean dataset mean: {mean_weight:.2f} mg")
        print(f"   New baseline to beat: {mae_baseline:.2f} mg MAE")
    
    return cleaned_df

if __name__ == "__main__":
    # Check required files
    required_files = ['biomass_outliers.csv', 'bin_results/bin_id_biomass_mapping.csv']
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ Required file not found: {file_path}")
            exit(1)
    
    print("✅ Required files found")
    
    # Create cleaned dataset
    cleaned_dataset = create_cleaned_dataset()
    
    print(f"\n🎉 Dataset cleaning completed!")
    print(f"   Ready for training on {len(cleaned_dataset)} clean BIN groups")