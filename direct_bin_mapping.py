#!/usr/bin/env python3
"""
Direct BIN Mapping Analysis - No bioscan-dataset package needed!
Works directly with CSV files using pandas only.
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

def load_biomass_data(biomass_file):
    """Load biomass data"""
    print(f" Loading biomass data from {biomass_file}")
    
    try:
        df = pd.read_csv(biomass_file, sep=',')
        print(f" Loaded {len(df):,} biomass records")
        
        # Rename columns to standard format
        if 'index' in df.columns and 'processid' not in df.columns:
            df['processid'] = df['index']
            print(" Renamed 'index' column to 'processid'")
        
        if 'weight_avg' in df.columns and 'weight' not in df.columns:
            df['weight'] = df['weight_avg']
            print(" Renamed 'weight_avg' column to 'weight'")
        
        # Clean data
        initial_len = len(df)
        df = df.dropna(subset=['weight'])
        print(f" Removed {initial_len - len(df)} records with missing weight")
        
        return df
        
    except Exception as e:
        print(f" Error loading biomass data: {e}")
        return None

def load_bioscan_metadata_direct(metadata_file):
    """Load BIOSCAN metadata directly with pandas"""
    print(f" Loading BIOSCAN metadata directly from {metadata_file}")
    
    if not os.path.exists(metadata_file):
        print(f" Metadata file not found: {metadata_file}")
        return None
    
    file_size = os.path.getsize(metadata_file)
    print(f" File size: {file_size / (1024**3):.2f} GB")
    
    try:
        # Load in chunks to handle large file
        print("⏳ Loading large dataset in chunks...")
        chunk_size = 100000
        chunks = []
        
        for i, chunk in enumerate(pd.read_csv(metadata_file, chunksize=chunk_size)):
            chunks.append(chunk)
            if (i + 1) % 10 == 0:
                print(f"   Loaded {(i + 1) * chunk_size:,} records...")
        
        print(" Concatenating chunks...")
        df = pd.concat(chunks, ignore_index=True)
        print(f" Loaded {len(df):,} BIOSCAN records")
        
        # Show column info
        print(f" Columns: {list(df.columns)}")
        
        # Check for BIN data
        if 'bin_uri' in df.columns:
            bin_count = df['bin_uri'].notna().sum()
            print(f" Found {bin_count:,} records with BIN IDs")
        elif 'dna_bin' in df.columns:
            bin_count = df['dna_bin'].notna().sum()
            print(f" Found {bin_count:,} records with BIN IDs")
        else:
            print(" No obvious BIN column found")
            print("Available columns:", list(df.columns)[:20])
        
        return df
        
    except Exception as e:
        print(f" Error loading BIOSCAN metadata: {e}")
        return None

def create_bin_mapping(bioscan_df, biomass_df, output_dir):
    """Create BIN to biomass mapping"""
    print("\n Creating BIN to biomass mapping...")
    
    # Find the BIN column
    bin_col = None
    for col in ['bin_uri', 'dna_bin', 'BIN']:
        if col in bioscan_df.columns:
            bin_col = col
            break
    
    if bin_col is None:
        print(" No BIN column found in BIOSCAN data")
        return False
    
    print(f" Using BIN column: {bin_col}")
    
    # Get biomass BIN IDs to filter BIOSCAN data first
    print(" Filtering BIOSCAN data to biomass BIN IDs only...")
    biomass_bins = set(biomass_df[bin_col].dropna().values)
    print(f" Looking for {len(biomass_bins):,} unique BIN IDs from biomass data")
    
    # Filter BIOSCAN data to only specimens with matching BIN IDs
    bioscan_filtered = bioscan_df[bioscan_df[bin_col].isin(biomass_bins)]
    print(f" Found {len(bioscan_filtered):,} BIOSCAN specimens with matching BIN IDs")
    
    # Now merge on BIN ID
    print(" Merging datasets by BIN ID...")
    merged = pd.merge(
        bioscan_filtered[['processid', bin_col, 'species', 'genus', 'family', 'order']],
        biomass_df[[bin_col, 'weight', 'processid']],
        on=bin_col,
        how='inner',
        suffixes=('_bioscan', '_biomass')
    )
    
    print(f" Found {len(merged):,} specimens with both BIOSCAN and biomass data")
    
    if len(merged) == 0:
        print(" No matching specimens found!")
        return False
    
    # Remove records without BIN IDs
    with_bins = merged.dropna(subset=[bin_col])
    print(f" {len(with_bins):,} specimens have BIN IDs")
    
    if len(with_bins) == 0:
        print(" No specimens with BIN IDs found!")
        return False
    
    # Create BIN statistics
    print(" Aggregating by BIN ID...")
    bin_stats = with_bins.groupby(bin_col).agg({
        'weight': ['count', 'mean', 'std', 'min', 'max'],
        'species': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0],
        'genus': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0],
        'family': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0],
        'order': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0],
        'processid_bioscan': list,
        'processid_biomass': list
    }).round(4)
    
    # Flatten column names
    bin_stats.columns = [
        'specimen_count', 'mean_weight', 'std_weight', 'min_weight', 'max_weight',
        'most_common_species', 'most_common_genus', 'most_common_family', 'most_common_order',
        'bioscan_processids', 'biomass_processids'
    ]
    
    bin_stats = bin_stats.reset_index()
    
    print(f" Created mapping for {len(bin_stats):,} unique BIN IDs")
    print(f" Average specimens per BIN: {bin_stats['specimen_count'].mean():.1f}")
    
    # Save results
    Path(output_dir).mkdir(exist_ok=True)
    
    # Save BIN mapping
    bin_file = os.path.join(output_dir, "bin_id_biomass_mapping.csv")
    bin_stats.to_csv(bin_file, index=False)
    print(f" Saved BIN mapping: {bin_file}")
    
    # Save summary stats
    stats_file = os.path.join(output_dir, "summary_statistics.txt")
    with open(stats_file, 'w') as f:
        f.write("DIRECT BIN MAPPING ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total BIOSCAN specimens: {len(bioscan_df):,}\n")
        f.write(f"Total biomass specimens: {len(biomass_df):,}\n")
        f.write(f"Specimens with both: {len(merged):,}\n")
        f.write(f"Specimens with BIN IDs: {len(with_bins):,}\n")
        f.write(f"Unique BIN IDs: {len(bin_stats):,}\n\n")
        
        f.write("Weight statistics:\n")
        f.write(f"  Mean: {biomass_df['weight'].mean():.4f}\n")
        f.write(f"  Std:  {biomass_df['weight'].std():.4f}\n")
        f.write(f"  Min:  {biomass_df['weight'].min():.4f}\n")
        f.write(f"  Max:  {biomass_df['weight'].max():.4f}\n")
        f.write(f"  Count: {len(biomass_df):,}\n\n")
        
        f.write("Top 10 BINs by specimen count:\n")
        top_bins = bin_stats.nlargest(10, 'specimen_count')
        for _, row in top_bins.iterrows():
            f.write(f"  {row[bin_col]}: {row['specimen_count']} specimens\n")
    
    print(f" Saved summary: {stats_file}")
    
    # Show preview
    print("\n BIN Mapping Preview (first 10 rows):")
    print(bin_stats.head(10).to_string())
    
    return True

def main():
    print(" DIRECT BIN MAPPING ANALYSIS (No bioscan-dataset package)")
    print("=" * 70)
    
    # File paths
    biomass_file = "5m_mass_metadata_avg.csv"
    metadata_file = "/home/vikramre/links/scratch/bioscan_data/bioscan5m/metadata/csv/BIOSCAN_5M_Insect_Dataset_metadata.csv"
    output_dir = "./bin_results"
    
    # Load data
    biomass_df = load_biomass_data(biomass_file)
    if biomass_df is None:
        sys.exit(1)
    
    bioscan_df = load_bioscan_metadata_direct(metadata_file)
    if bioscan_df is None:
        sys.exit(1)
    
    # Create mapping
    success = create_bin_mapping(bioscan_df, biomass_df, output_dir)
    
    if success:
        print(f"\n SUCCESS! BIN mapping analysis completed")
        print(f" Results saved to {output_dir}/")
    else:
        print(f"\n FAILED! Check the error messages above")
        sys.exit(1)

if __name__ == "__main__":
    main()