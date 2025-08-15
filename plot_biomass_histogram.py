#!/usr/bin/env python3
"""
Plot histogram of all biomass values from the cleaned dataset.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    print("📊 BIOMASS HISTOGRAM ANALYSIS")
    print("=" * 40)
    
    # Load cleaned biomass data (same path as in run_kmeans_cv_job.sh)
    data_path = "bin_results/bin_id_biomass_mapping_cleaned.csv"
    
    try:
        df = pd.read_csv(data_path)
        print(f"✅ Loaded {len(df)} biomass samples")
        
        # Get biomass column (check what columns exist)
        print(f"📋 Columns: {list(df.columns)}")
        
        # Use the correct biomass column name as seen in the training pipeline
        biomass_col = 'mean_weight'
        if biomass_col not in df.columns:
            print(f"❌ Column '{biomass_col}' not found in data")
            print("Available columns:", list(df.columns))
            return
            
        biomass_values = df[biomass_col].dropna()
        
        print(f"📊 Biomass Statistics:")
        print(f"   Count: {len(biomass_values)}")
        print(f"   Mean: {biomass_values.mean():.3f} mg")
        print(f"   Median: {biomass_values.median():.3f} mg")
        print(f"   Std: {biomass_values.std():.3f} mg")
        print(f"   Min: {biomass_values.min():.3f} mg")
        print(f"   Max: {biomass_values.max():.3f} mg")
        
        # Create histogram
        plt.figure(figsize=(12, 8))
        
        # Main histogram
        plt.subplot(2, 2, 1)
        plt.hist(biomass_values, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        plt.title('Biomass Distribution (All Values)')
        plt.xlabel('Biomass (mg)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Log scale histogram
        plt.subplot(2, 2, 2)
        plt.hist(biomass_values, bins=50, alpha=0.7, color='forestgreen', edgecolor='black')
        plt.yscale('log')
        plt.title('Biomass Distribution (Log Scale)')
        plt.xlabel('Biomass (mg)')
        plt.ylabel('Frequency (log scale)')
        plt.grid(True, alpha=0.3)
        
        # Box plot
        plt.subplot(2, 2, 3)
        plt.boxplot(biomass_values, vert=True)
        plt.title('Biomass Box Plot')
        plt.ylabel('Biomass (mg)')
        plt.grid(True, alpha=0.3)
        
        # Cumulative distribution
        plt.subplot(2, 2, 4)
        sorted_values = np.sort(biomass_values)
        cumulative = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
        plt.plot(sorted_values, cumulative, color='red', linewidth=2)
        plt.title('Cumulative Distribution')
        plt.xlabel('Biomass (mg)')
        plt.ylabel('Cumulative Probability')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_file = 'biomass_histogram_analysis.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"💾 Histogram saved as: {output_file}")
        
        # Show percentiles
        print(f"\n📊 Percentiles:")
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            value = np.percentile(biomass_values, p)
            print(f"   {p:2d}th percentile: {value:.3f} mg")
        
        plt.show()
        
    except FileNotFoundError:
        print(f"❌ Data file not found: {data_path}")
        print("Available files:")
        import os
        if os.path.exists("bin_results"):
            for f in os.listdir("bin_results"):
                print(f"   bin_results/{f}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()