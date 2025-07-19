"""
BIN ID to Biomass Mapping and Data Gap Analysis

This script creates several useful files:
1. BIN ID to biomass mapping (for specimens with both data)
2. Specimens with images but no biomass data
3. Specimens with biomass but no images  
4. Summary statistics

Usage:
    python create_bin_mapping.py --biomass_file your_biomass.tsv --bioscan_root ./bioscan_data
"""

import pandas as pd
import numpy as np
import os
import argparse
from typing import Dict, List, Tuple, Set
from pathlib import Path

# Import BIOSCAN dataset tools
try:
    from bioscan_dataset import BIOSCAN5M
    BIOSCAN_AVAILABLE = True
except ImportError:
    print("⚠️ bioscan-dataset not available, working with biomass data only")
    BIOSCAN_AVAILABLE = False

def load_biomass_data(biomass_file: str) -> pd.DataFrame:
    """Load and clean biomass data"""
    print(f"📊 Loading biomass data from {biomass_file}")
    
    try:
        df = pd.read_csv(biomass_file, sep=',')
        print(f"✅ Loaded {len(df)} biomass records")
        
        # Rename columns to match expected format
        if 'index' in df.columns and 'processid' not in df.columns:
            df['processid'] = df['index']
            print("🔧 Renamed 'index' column to 'processid'")
        
        if 'weight_avg' in df.columns and 'weight' not in df.columns:
            df['weight'] = df['weight_avg']
            print("🔧 Renamed 'weight_avg' column to 'weight'")
        
        # Extract processid from image_file if needed
        if 'processid' not in df.columns and 'image_file' in df.columns:
            df['processid'] = df['image_file'].str.replace('.jpg', '').str.replace('.JPG', '')
            print("🔧 Extracted processid from image_file column")
        
        # Check required columns
        required_cols = ['processid', 'weight']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            print(f"Available columns: {list(df.columns)}")
            return None
            
        # Remove rows with missing weight data
        initial_len = len(df)
        df = df.dropna(subset=['weight'])
        print(f"🧹 Removed {initial_len - len(df)} records with missing weight data")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading biomass data: {e}")
        return None

def load_bioscan_metadata(bioscan_root: str, download: bool = False) -> pd.DataFrame:
    """Load BIOSCAN metadata without downloading images"""
    print(f"🔍 Loading BIOSCAN metadata from {bioscan_root}")
    
    if not BIOSCAN_AVAILABLE:
        print("❌ bioscan-dataset package not available")
        return None
    
    try:
        # Try to load metadata file directly
        metadata_path = os.path.join(bioscan_root, "bioscan5m", "metadata", "csv", "BIOSCAN_5M_Insect_Dataset_metadata.csv")
        
        if not os.path.exists(metadata_path):
            print(f"📥 Metadata file not found, attempting to download...")
            if download:
                # Create minimal dataset just to download metadata
                try:
                    dataset = BIOSCAN5M(root=bioscan_root, download=True, modality=())
                    # Get metadata from the dataset
                    df = dataset.metadata
                    print(f"✅ Downloaded and loaded {len(df)} BIOSCAN records")
                except Exception as e:
                    print(f"⚠️  Download failed: {e}")
                    return None
            else:
                print(f"❌ Metadata file not found: {metadata_path}")
                print("💡 Run with --download flag to download metadata")
                return None
        else:
            # Load existing metadata
            df = pd.read_csv(metadata_path)
            print(f"✅ Loaded {len(df)} BIOSCAN records")
        
        # Check for BIN data
        if 'dna_bin' in df.columns:
            bin_count = df['dna_bin'].notna().sum()
            print(f"🧬 Found {bin_count} records with BIN IDs")
        else:
            print("❌ No dna_bin column found in BIOSCAN data")
            
        return df
        
    except Exception as e:
        print(f"❌ Error loading BIOSCAN metadata: {e}")
        return None

def create_bin_biomass_mapping(bioscan_df: pd.DataFrame, biomass_df: pd.DataFrame) -> pd.DataFrame:
    """Create mapping of BIN IDs to biomass data"""
    print("\n🔗 Creating BIN ID to biomass mapping...")
    
    # Merge datasets on processid
    merged = pd.merge(
        bioscan_df[['processid', 'dna_bin', 'species', 'genus', 'family', 'order']],
        biomass_df[['processid', 'weight', 'area_fraction', 'image_measurement_value', 'scale_factor']],
        on='processid',
        how='inner'
    )
    
    print(f"🔍 Found {len(merged)} specimens with both BIOSCAN and biomass data")
    
    if len(merged) == 0:
        print("❌ No matching specimens found!")
        print("💡 Check that processid columns match between datasets")
        return pd.DataFrame()
    
    # Remove records without BIN IDs
    with_bins = merged.dropna(subset=['dna_bin'])
    print(f"🧬 {len(with_bins)} specimens have BIN IDs")
    
    if len(with_bins) == 0:
        print("❌ No specimens with BIN IDs found!")
        return pd.DataFrame()
    
    # Aggregate by BIN ID (multiple specimens per BIN)
    bin_stats = with_bins.groupby('dna_bin').agg({
        'weight': ['count', 'mean', 'std', 'min', 'max'],
        'area_fraction': 'mean',
        'image_measurement_value': 'mean', 
        'scale_factor': 'mean',
        'species': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0],
        'genus': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0],
        'family': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0],
        'order': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0],
        'processid': lambda x: list(x)  # Keep all processids for reference
    }).round(4)
    
    # Flatten column names
    bin_stats.columns = [
        'specimen_count', 'mean_weight', 'std_weight', 'min_weight', 'max_weight',
        'mean_area_fraction', 'mean_image_measurement', 'mean_scale_factor',
        'most_common_species', 'most_common_genus', 'most_common_family', 'most_common_order',
        'processids'
    ]
    
    # Reset index to make dna_bin a column
    bin_stats = bin_stats.reset_index()
    
    print(f"📊 Created mapping for {len(bin_stats)} unique BIN IDs")
    print(f"📈 Average specimens per BIN: {bin_stats['specimen_count'].mean():.1f}")
    
    return bin_stats

def find_images_without_biomass(bioscan_df: pd.DataFrame, biomass_df: pd.DataFrame) -> pd.DataFrame:
    """Find specimens with images but no biomass data"""
    print("\n🖼️  Finding specimens with images but no biomass...")
    
    # Get processids with biomass data
    biomass_processids = set(biomass_df['processid'])
    
    # Find BIOSCAN specimens without biomass
    no_biomass = bioscan_df[~bioscan_df['processid'].isin(biomass_processids)].copy()
    
    print(f"📊 Found {len(no_biomass)} specimens with images but no biomass data")
    
    # Add useful columns
    if len(no_biomass) > 0:
        result = no_biomass[['processid', 'dna_bin', 'species', 'genus', 'family', 'order', 
                           'image_measurement_value', 'area_fraction', 'scale_factor', 
                           'split', 'country']].copy()
        
        # Add estimated weight from BIN mapping if available
        result['has_bin_id'] = result['dna_bin'].notna()
        
        return result
    
    return pd.DataFrame()

def find_biomass_without_images(bioscan_df: pd.DataFrame, biomass_df: pd.DataFrame) -> pd.DataFrame:
    """Find specimens with biomass but no images"""
    print("\n⚖️  Finding specimens with biomass but no images...")
    
    # Get processids with images
    bioscan_processids = set(bioscan_df['processid'])
    
    # Find biomass specimens without images
    no_images = biomass_df[~biomass_df['processid'].isin(bioscan_processids)].copy()
    
    print(f"📊 Found {len(no_images)} specimens with biomass but no images")
    
    return no_images

def generate_summary_stats(bioscan_df: pd.DataFrame, biomass_df: pd.DataFrame, 
                          bin_mapping: pd.DataFrame, no_biomass: pd.DataFrame, 
                          no_images: pd.DataFrame) -> Dict:
    """Generate comprehensive summary statistics"""
    
    stats = {
        'total_bioscan_specimens': len(bioscan_df),
        'total_biomass_specimens': len(biomass_df),
        'specimens_with_both': len(bin_mapping) if not bin_mapping.empty else 0,
        'specimens_images_only': len(no_biomass),
        'specimens_biomass_only': len(no_images),
        'unique_bin_ids': bioscan_df['dna_bin'].nunique(),
        'bin_ids_with_biomass': len(bin_mapping) if not bin_mapping.empty else 0,
        'bioscan_splits': bioscan_df['split'].value_counts().to_dict() if 'split' in bioscan_df.columns else {},
        'weight_stats': {
            'mean': biomass_df['weight'].mean(),
            'std': biomass_df['weight'].std(), 
            'min': biomass_df['weight'].min(),
            'max': biomass_df['weight'].max(),
            'count': len(biomass_df)
        } if 'weight' in biomass_df.columns else {}
    }
    
    return stats

def save_results(bin_mapping: pd.DataFrame, no_biomass: pd.DataFrame, 
                no_images: pd.DataFrame, stats: Dict, output_dir: str = "./output"):
    """Save all results to files"""
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    print(f"\n💾 Saving results to {output_dir}/")
    
    # Save BIN to biomass mapping
    if not bin_mapping.empty:
        bin_file = os.path.join(output_dir, "bin_id_biomass_mapping.csv")
        bin_mapping.to_csv(bin_file, index=False)
        print(f"✅ Saved BIN mapping: {bin_file}")
    
    # Save specimens with images but no biomass
    if not no_biomass.empty:
        images_file = os.path.join(output_dir, "specimens_images_no_biomass.csv")
        no_biomass.to_csv(images_file, index=False)
        print(f"✅ Saved images without biomass: {images_file}")
    
    # Save specimens with biomass but no images
    if not no_images.empty:
        biomass_file = os.path.join(output_dir, "specimens_biomass_no_images.csv")
        no_images.to_csv(biomass_file, index=False)
        print(f"✅ Saved biomass without images: {biomass_file}")
    
    # Save summary statistics
    stats_file = os.path.join(output_dir, "data_summary_statistics.txt")
    with open(stats_file, 'w') as f:
        f.write("BIOSCAN-BIOMASS DATA ANALYSIS SUMMARY\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"Total BIOSCAN specimens: {stats['total_bioscan_specimens']:,}\n")
        f.write(f"Total biomass specimens: {stats['total_biomass_specimens']:,}\n")
        f.write(f"Specimens with both: {stats['specimens_with_both']:,}\n")
        f.write(f"Specimens with images only: {stats['specimens_images_only']:,}\n")
        f.write(f"Specimens with biomass only: {stats['specimens_biomass_only']:,}\n\n")
        
        f.write(f"Unique BIN IDs in BIOSCAN: {stats['unique_bin_ids']:,}\n")
        f.write(f"BIN IDs with biomass data: {stats['bin_ids_with_biomass']:,}\n\n")
        
        if stats['weight_stats']:
            ws = stats['weight_stats']
            f.write(f"Weight statistics:\n")
            f.write(f"  Mean: {ws['mean']:.4f}\n")
            f.write(f"  Std:  {ws['std']:.4f}\n") 
            f.write(f"  Min:  {ws['min']:.4f}\n")
            f.write(f"  Max:  {ws['max']:.4f}\n")
            f.write(f"  Count: {ws['count']:,}\n\n")
        
        if stats['bioscan_splits']:
            f.write("BIOSCAN data splits:\n")
            for split, count in stats['bioscan_splits'].items():
                f.write(f"  {split}: {count:,}\n")
    
    print(f"✅ Saved summary statistics: {stats_file}")

def main():
    parser = argparse.ArgumentParser(description="Create BIN ID to biomass mapping and data gap analysis")
    parser.add_argument("--biomass_file", required=True, help="Path to biomass TSV file")
    parser.add_argument("--bioscan_root", default="./bioscan_data", help="BIOSCAN dataset root directory")
    parser.add_argument("--output_dir", default="./output", help="Output directory for results")
    parser.add_argument("--download", action="store_true", help="Download BIOSCAN metadata if missing")
    
    args = parser.parse_args()
    
    print("🧬 BIN ID TO BIOMASS MAPPING ANALYSIS")
    print("="*60)
    
    # Load data
    biomass_df = load_biomass_data(args.biomass_file)
    if biomass_df is None:
        return
        
    bioscan_df = load_bioscan_metadata(args.bioscan_root, args.download)
    if bioscan_df is None:
        return
    
    # Create mappings and find gaps
    bin_mapping = create_bin_biomass_mapping(bioscan_df, biomass_df)
    no_biomass = find_images_without_biomass(bioscan_df, biomass_df)
    no_images = find_biomass_without_images(bioscan_df, biomass_df)
    
    # Generate statistics
    stats = generate_summary_stats(bioscan_df, biomass_df, bin_mapping, no_biomass, no_images)
    
    # Save results
    save_results(bin_mapping, no_biomass, no_images, stats, args.output_dir)
    
    print(f"\n🎯 ANALYSIS COMPLETE!")
    print("="*60)
    print(f"📊 {stats['specimens_with_both']:,} specimens have both images and biomass")
    print(f"🖼️  {stats['specimens_images_only']:,} specimens have images but no biomass")
    print(f"⚖️  {stats['specimens_biomass_only']:,} specimens have biomass but no images")
    print(f"🧬 {stats['bin_ids_with_biomass']:,} BIN IDs have biomass data")
    print(f"\n💾 Results saved to {args.output_dir}/")

if __name__ == "__main__":
    main()