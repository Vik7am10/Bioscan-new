#!/usr/bin/env python3
"""
Create a temporary dataset with values above 100mg removed for histogram analysis.
"""

import pandas as pd
import numpy as np

def create_filtered_dataset():
    print("🔧 Creating Filtered Dataset (removing values > 100mg)")
    print("=" * 60)
    
    # Load cleaned dataset
    df = pd.read_csv('bin_results/bin_id_biomass_mapping_cleaned.csv')
    print(f"📁 Original cleaned dataset: {len(df)} BIN groups")
    
    # Show current statistics
    weights = df['mean_weight']
    print(f"📊 Current statistics:")
    print(f"   Range: {weights.min():.3f} to {weights.max():.3f} mg")
    print(f"   Mean: {weights.mean():.3f} mg")
    print(f"   Values > 100mg: {(weights > 100).sum()}")
    
    # Filter out values > 100mg
    filtered_df = df[df['mean_weight'] <= 100].copy()
    print(f"\n🔍 After filtering (≤ 100mg): {len(filtered_df)} BIN groups")
    print(f"   Removed: {len(df) - len(filtered_df)} BIN groups")
    
    # Show new statistics
    new_weights = filtered_df['mean_weight']
    print(f"\n📊 New statistics:")
    print(f"   Range: {new_weights.min():.3f} to {new_weights.max():.3f} mg")
    print(f"   Mean: {new_weights.mean():.3f} mg")
    print(f"   Median: {new_weights.median():.3f} mg")
    print(f"   95th percentile: {new_weights.quantile(0.95):.3f} mg")
    print(f"   99th percentile: {new_weights.quantile(0.99):.3f} mg")
    
    # Save temporary filtered dataset
    output_file = 'bin_results/bin_id_biomass_mapping_filtered_100mg.csv'
    filtered_df.to_csv(output_file, index=False)
    print(f"\n💾 Saved filtered dataset to: {output_file}")
    
    return output_file

if __name__ == "__main__":
    create_filtered_dataset()