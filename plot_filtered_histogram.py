#!/usr/bin/env python3
"""
Plot histogram of biomass values with the filtered dataset (≤ 100mg).
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    print("📊 FILTERED BIOMASS HISTOGRAM (≤ 100mg)")
    print("=" * 50)
    
    # Check if filtered dataset exists, if not create it
    filtered_path = "bin_results/bin_id_biomass_mapping_filtered_100mg.csv"
    
    try:
        df = pd.read_csv(filtered_path)
        print(f"✅ Loaded filtered dataset: {len(df)} samples")
    except FileNotFoundError:
        print("🔧 Filtered dataset not found, creating it first...")
        from create_filtered_dataset import create_filtered_dataset
        filtered_path = create_filtered_dataset()
        df = pd.read_csv(filtered_path)
        print(f"✅ Created and loaded filtered dataset: {len(df)} samples")
    
    # Get biomass values
    biomass_values = df['mean_weight'].dropna()
    
    print(f"\n📊 Filtered Biomass Statistics:")
    print(f"   Count: {len(biomass_values)}")
    print(f"   Mean: {biomass_values.mean():.3f} mg")
    print(f"   Median: {biomass_values.median():.3f} mg")
    print(f"   Std: {biomass_values.std():.3f} mg")
    print(f"   Min: {biomass_values.min():.3f} mg")
    print(f"   Max: {biomass_values.max():.3f} mg")
    
    # Create histogram with better resolution for smaller range
    plt.figure(figsize=(15, 10))
    
    # Main histogram
    plt.subplot(2, 3, 1)
    plt.hist(biomass_values, bins=60, alpha=0.7, color='steelblue', edgecolor='black')
    plt.title('Filtered Biomass Distribution (≤ 100mg)')
    plt.xlabel('Biomass (mg)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Log scale histogram
    plt.subplot(2, 3, 2)
    plt.hist(biomass_values, bins=60, alpha=0.7, color='forestgreen', edgecolor='black')
    plt.yscale('log')
    plt.title('Filtered Distribution (Log Scale)')
    plt.xlabel('Biomass (mg)')
    plt.ylabel('Frequency (log scale)')
    plt.grid(True, alpha=0.3)
    
    # Zoomed in on lower values (0-10mg)
    plt.subplot(2, 3, 3)
    low_values = biomass_values[biomass_values <= 10]
    plt.hist(low_values, bins=50, alpha=0.7, color='orange', edgecolor='black')
    plt.title('Low Range Detail (≤ 10mg)')
    plt.xlabel('Biomass (mg)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Box plot
    plt.subplot(2, 3, 4)
    plt.boxplot(biomass_values, vert=True)
    plt.title('Filtered Biomass Box Plot')
    plt.ylabel('Biomass (mg)')
    plt.grid(True, alpha=0.3)
    
    # Cumulative distribution
    plt.subplot(2, 3, 5)
    sorted_values = np.sort(biomass_values)
    cumulative = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
    plt.plot(sorted_values, cumulative, color='red', linewidth=2)
    plt.title('Cumulative Distribution')
    plt.xlabel('Biomass (mg)')
    plt.ylabel('Cumulative Probability')
    plt.grid(True, alpha=0.3)
    
    # High range detail (10-100mg)
    plt.subplot(2, 3, 6)
    high_values = biomass_values[(biomass_values > 10) & (biomass_values <= 100)]
    if len(high_values) > 0:
        plt.hist(high_values, bins=30, alpha=0.7, color='purple', edgecolor='black')
        plt.title('High Range Detail (10-100mg)')
        plt.xlabel('Biomass (mg)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No values in\n10-100mg range', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('High Range Detail (10-100mg)')
    
    plt.tight_layout()
    
    # Save plot
    output_file = 'biomass_histogram_filtered_100mg.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n💾 Histogram saved as: {output_file}")
    
    # Show percentiles
    print(f"\n📊 Percentiles:")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        value = np.percentile(biomass_values, p)
        print(f"   {p:2d}th percentile: {value:.3f} mg")
    
    plt.show()

if __name__ == "__main__":
    main()