#!/usr/bin/env python3
"""
Filter outliers from biomass dataset and calculate baseline MAPE/MAE metrics
"""

import pandas as pd
import numpy as np
import os
import argparse
from pathlib import Path


def calculate_baseline_metrics(actual, predicted):
    """Calculate MAPE and MAE metrics."""
    
    # Mean Absolute Error
    mae = np.mean(np.abs(actual - predicted))
    
    # Mean Absolute Percentage Error (handle division by zero)
    # Add small epsilon to avoid division by zero
    epsilon = 1e-8
    mape = np.mean(np.abs((actual - predicted) / (actual + epsilon))) * 100
    
    # Alternative MAPE calculation (symmetric)
    smape = np.mean(2 * np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted) + epsilon)) * 100
    
    return mae, mape, smape


def filter_outliers_and_baseline(use_specimens=True, max_biomass=None):
    """Filter outliers and calculate baseline performance.
    
    Args:
        use_specimens (bool): If True, use individual specimens. If False, average by BIN.
        max_biomass (float): If provided, filter out samples above this biomass (in mg).
    """
    
    mode = "Individual Specimens" if use_specimens else "BIN-Averaged"
    biomass_filter = f" (≤{max_biomass}mg)" if max_biomass else ""
    print(f"🔍 Filtering Outliers and Calculating Baseline Performance ({mode}{biomass_filter})")
    print("=" * 60)
    
    # Load the analysis results
    if not os.path.exists('biomass_feature_analysis.csv'):
        print("❌ Error: biomass_feature_analysis.csv not found. Run biomass_outlier_analysis.py first.")
        return
    
    if not os.path.exists('biomass_outliers.csv'):
        print("❌ Error: biomass_outliers.csv not found. Run biomass_outlier_analysis.py first.")
        return
    
    print("📊 Loading analysis data...")
    df_full = pd.read_csv('biomass_feature_analysis.csv')
    df_outliers = pd.read_csv('biomass_outliers.csv')
    
    print(f"   Full dataset: {len(df_full)} samples")
    print(f"   Outliers: {len(df_outliers)} samples ({len(df_outliers)/len(df_full)*100:.1f}%)")
    
    # Create filtered dataset (remove outliers)
    outlier_processids = set(df_outliers['processid'])
    df_filtered = df_full[~df_full['processid'].isin(outlier_processids)].copy()
    
    print(f"   After outlier removal: {len(df_filtered)} samples")
    print(f"   Removed outliers: {len(df_full) - len(df_filtered)} samples")
    
    # Apply biomass filter if specified
    if max_biomass is not None:
        df_before_biomass_filter = df_filtered.copy()
        df_filtered = df_filtered[df_filtered['weight'] <= max_biomass].copy()
        
        print(f"   After biomass filter (≤{max_biomass}mg): {len(df_filtered)} samples")
        print(f"   Removed high biomass: {len(df_before_biomass_filter) - len(df_filtered)} samples")
        
        # Show biomass distribution
        high_biomass_samples = df_before_biomass_filter[df_before_biomass_filter['weight'] > max_biomass]
        if len(high_biomass_samples) > 0:
            print(f"   High biomass range: {high_biomass_samples['weight'].min():.1f} - {high_biomass_samples['weight'].max():.1f} mg")
            print(f"   Kept biomass range: {df_filtered['weight'].min():.1f} - {df_filtered['weight'].max():.1f} mg")
    
    # If using BIN-averaged approach, aggregate by BIN
    if not use_specimens:
        print(f"\n📊 Aggregating by BIN (averaging specimens per species)...")
        
        # Average specimens within each BIN, preserving all columns
        agg_dict = {}
        for col in df_filtered.columns:
            if col == 'bin_id':
                continue  # This is the groupby key
            elif col in ['processid']:
                agg_dict[col] = 'first'  # Take first processid as representative
            elif col in ['specimen_count']:
                agg_dict[col] = 'first'  # Keep original specimen count
            else:
                agg_dict[col] = 'mean'  # Average numerical columns
        
        df_filtered_agg = df_filtered.groupby('bin_id').agg(agg_dict).reset_index()
        
        print(f"   BIN-averaged dataset: {len(df_filtered_agg)} BIN groups")
        df_analysis_data = df_filtered_agg
        weight_col = 'weight'
        area_col = 'pixel_area'
    else:
        df_analysis_data = df_filtered
        weight_col = 'weight'
        area_col = 'pixel_area'
    
    # Dataset statistics comparison
    print(f"\n📈 Dataset Statistics Comparison:")
    print(f"{'Metric':<20} {'Full Dataset':<15} {'Filtered':<15} {'Change':<15}")
    print("-" * 65)
    
    metrics = [
        ('Count', len(df_full), len(df_analysis_data)),
        ('Mean (mg)', df_full['weight'].mean(), df_analysis_data[weight_col].mean()),
        ('Median (mg)', df_full['weight'].median(), df_analysis_data[weight_col].median()),
        ('Std (mg)', df_full['weight'].std(), df_analysis_data[weight_col].std()),
        ('Max (mg)', df_full['weight'].max(), df_analysis_data[weight_col].max()),
        ('95th perc (mg)', df_full['weight'].quantile(0.95), df_analysis_data[weight_col].quantile(0.95))
    ]
    
    for metric_name, full_val, filt_val in metrics:
        if metric_name == 'Count':
            change = f"-{full_val - filt_val}"
        else:
            change_pct = ((filt_val - full_val) / full_val) * 100 if full_val != 0 else 0
            change = f"{change_pct:+.1f}%"
        
        if isinstance(full_val, (int, float)) and full_val > 1:
            print(f"{metric_name:<20} {full_val:<15.2f} {filt_val:<15.2f} {change:<15}")
        else:
            print(f"{metric_name:<20} {full_val:<15} {filt_val:<15} {change:<15}")
    
    # Calculate baseline predictions using different methods
    print(f"\n🎯 Baseline Prediction Methods:")
    print("=" * 50)
    
    # Method 1: Simple mean prediction
    mean_pred = np.full(len(df_analysis_data), df_analysis_data[weight_col].mean())
    mae_mean, mape_mean, smape_mean = calculate_baseline_metrics(df_analysis_data[weight_col].values, mean_pred)
    
    # Method 2: Median prediction
    median_pred = np.full(len(df_analysis_data), df_analysis_data[weight_col].median())
    mae_median, mape_median, smape_median = calculate_baseline_metrics(df_analysis_data[weight_col].values, median_pred)
    
    # Method 3: Linear scaling from pixel area (simple regression)
    # Use numpy for simple linear regression to avoid threading issues
    X = df_analysis_data[area_col].values.reshape(-1, 1)
    y = df_analysis_data[weight_col].values
    
    # Simple linear regression: y = ax + b
    X_mean = np.mean(X)
    y_mean = np.mean(y)
    
    # Calculate slope (a) and intercept (b)
    numerator = np.sum((X.flatten() - X_mean) * (y - y_mean))
    denominator = np.sum((X.flatten() - X_mean) ** 2)
    
    if denominator != 0:
        slope = numerator / denominator
        intercept = y_mean - slope * X_mean
        
        linear_pred = slope * X.flatten() + intercept
        # Ensure predictions are non-negative
        linear_pred = np.maximum(linear_pred, 0.001)
        
        mae_linear, mape_linear, smape_linear = calculate_baseline_metrics(y, linear_pred)
    else:
        mae_linear = mape_linear = smape_linear = float('inf')
        slope = intercept = 0
    
    # Method 4: Square root scaling (biological scaling law)
    sqrt_area = np.sqrt(df_analysis_data[area_col].values)
    sqrt_mean = np.mean(sqrt_area)
    
    numerator_sqrt = np.sum((sqrt_area - sqrt_mean) * (y - y_mean))
    denominator_sqrt = np.sum((sqrt_area - sqrt_mean) ** 2)
    
    if denominator_sqrt != 0:
        slope_sqrt = numerator_sqrt / denominator_sqrt
        intercept_sqrt = y_mean - slope_sqrt * sqrt_mean
        
        sqrt_pred = slope_sqrt * sqrt_area + intercept_sqrt
        sqrt_pred = np.maximum(sqrt_pred, 0.001)
        
        mae_sqrt, mape_sqrt, smape_sqrt = calculate_baseline_metrics(y, sqrt_pred)
    else:
        mae_sqrt = mape_sqrt = smape_sqrt = float('inf')
        slope_sqrt = intercept_sqrt = 0
    
    # Display results
    results = [
        ('Mean Prediction', mae_mean, mape_mean, smape_mean),
        ('Median Prediction', mae_median, mape_median, smape_median),
        ('Linear Scaling', mae_linear, mape_linear, smape_linear),
        ('Sqrt Scaling', mae_sqrt, mape_sqrt, smape_sqrt)
    ]
    
    print(f"{'Method':<20} {'MAE (mg)':<12} {'MAPE (%)':<12} {'SMAPE (%)':<12}")
    print("-" * 56)
    
    best_mae = float('inf')
    best_method = ""
    
    for method, mae, mape, smape in results:
        print(f"{method:<20} {mae:<12.3f} {mape:<12.1f} {smape:<12.1f}")
        if mae < best_mae:
            best_mae = mae
            best_method = method
    
    print(f"\n🏆 Best Baseline Method: {best_method} (MAE: {best_mae:.3f} mg)")
    
    # Save filtered dataset
    print(f"\n💾 Saving Results...")
    
    # Create cleaned dataset for training (use the analysis data, not original filtered)
    df_filtered_clean = df_analysis_data.copy()
    
    # Add baseline predictions to the dataset
    df_filtered_clean['mean_baseline'] = mean_pred
    df_filtered_clean['median_baseline'] = median_pred
    if mae_linear != float('inf'):
        df_filtered_clean['linear_baseline'] = linear_pred
    if mae_sqrt != float('inf'):
        df_filtered_clean['sqrt_baseline'] = sqrt_pred
    
    # Save cleaned dataset with appropriate suffix
    suffix = f"_max{int(max_biomass)}mg" if max_biomass else ""
    mode_suffix = "_bins" if not use_specimens else "_specimens"
    
    output_file = f'bin_results/bin_id_biomass_mapping_cleaned_final{suffix}{mode_suffix}.csv'
    
    # Create summary for BIN-level data (average across specimens in each BIN)
    bin_agg_dict = {}
    for col in df_filtered_clean.columns:
        if col == 'bin_id':
            continue  # This is the groupby key
        elif col in ['processid']:
            bin_agg_dict[col] = 'first'  # Take first processid as representative
        elif col in ['specimen_count']:
            bin_agg_dict[col] = 'first'  # Keep original specimen count
        else:
            bin_agg_dict[col] = 'mean'  # Average numerical columns
    
    bin_summary = df_filtered_clean.groupby('bin_id').agg(bin_agg_dict).reset_index()
    
    # Rename key columns for clarity
    if 'weight' in bin_summary.columns:
        bin_summary.rename(columns={'weight': 'mean_weight'}, inplace=True)
    if 'pixel_area' in bin_summary.columns:
        bin_summary.rename(columns={'pixel_area': 'mean_pixel_area'}, inplace=True)
    bin_summary.to_csv(output_file, index=False)
    
    # Save detailed specimen-level data (save original filtered data for specimens mode)
    if use_specimens:
        specimen_file = f'biomass_specimens_cleaned{suffix}{mode_suffix}.csv'
        df_filtered.to_csv(specimen_file, index=False)
        print(f"   Cleaned specimen dataset: {specimen_file} ({len(df_filtered)} specimens)")
    
    print(f"   Cleaned analysis dataset: {output_file} ({len(df_filtered_clean)} samples)")
    if not use_specimens:
        print(f"   (BIN-averaged from {len(df_filtered)} original specimens)")
    
    # Save baseline results
    baseline_results = pd.DataFrame({
        'method': [r[0] for r in results],
        'mae': [r[1] for r in results],
        'mape': [r[2] for r in results],
        'smape': [r[3] for r in results]
    })
    baseline_file = f'baseline_results{suffix}{mode_suffix}.csv'
    baseline_results.to_csv(baseline_file, index=False)
    
    print(f"   Baseline results: {baseline_file}")
    
    # Print summary for model training
    print(f"\n🚀 Ready for Model Training:")
    if use_specimens:
        print(f"   Training dataset: {len(df_filtered_clean)} specimens ({len(bin_summary)} BIN groups)")
        print(f"   Mean biomass: {df_filtered_clean['weight'].mean():.2f} ± {df_filtered_clean['weight'].std():.2f} mg")
        print(f"   Biomass range: [{df_filtered_clean['weight'].min():.2f}, {df_filtered_clean['weight'].max():.2f}] mg")
    else:
        print(f"   Training dataset: {len(df_analysis_data)} BIN groups (species-averaged)")
        print(f"   Mean biomass: {df_analysis_data[weight_col].mean():.2f} ± {df_analysis_data[weight_col].std():.2f} mg")
        print(f"   Biomass range: [{df_analysis_data[weight_col].min():.2f}, {df_analysis_data[weight_col].max():.2f}] mg")
    
    print(f"   Target baseline to beat: {best_mae:.3f} mg MAE")
    
    return df_filtered_clean, bin_summary, baseline_results


def main():
    """Main function with CLI argument parsing."""
    parser = argparse.ArgumentParser(description="Filter outliers and calculate baseline metrics")
    parser.add_argument('--mode', choices=['specimens', 'bins'], default='specimens',
                       help='Use individual specimens or BIN-averaged data (default: specimens)')
    parser.add_argument('--max-biomass', type=float, default=None,
                       help='Filter out samples above this biomass threshold (in mg)')
    parser.add_argument('--output-suffix', default='',
                       help='Suffix to add to output files')
    
    args = parser.parse_args()
    
    use_specimens = (args.mode == 'specimens')
    
    df_filtered, bin_summary, baseline_results = filter_outliers_and_baseline(
        use_specimens=use_specimens, 
        max_biomass=args.max_biomass
    )
    
    filter_info = f" (≤{args.max_biomass}mg)" if args.max_biomass else ""
    print(f"\n✅ Analysis complete! Mode: {args.mode}{filter_info}")


if __name__ == "__main__":
    main()