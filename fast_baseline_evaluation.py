#!/usr/bin/env python3
"""
Fast baseline evaluation using biomass values directly (no image loading).
"""

import os
import json
import numpy as np
import pandas as pd
import random


def calculate_metrics(predictions, targets, metric_name=""):
    """Calculate comprehensive metrics including MAPE."""
    
    # Basic metrics
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(mse)
    
    # MAPE (Mean Average Percentage Error)
    # Handle division by zero by adding small epsilon
    epsilon = 1e-8
    mape = np.mean(np.abs((targets - predictions) / (targets + epsilon))) * 100
    
    # Alternative MAPE calculation for very small values
    # Use symmetric MAPE which is more robust for values near zero
    smape = np.mean(2 * np.abs(predictions - targets) / (np.abs(predictions) + np.abs(targets) + epsilon)) * 100
    
    # R² coefficient of determination
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # Median metrics (more robust to outliers)
    median_ae = np.median(np.abs(predictions - targets))
    
    print(f"\n {metric_name} Metrics:")
    print(f"   MSE: {mse:.2f} mg²")
    print(f"   MAE: {mae:.2f} mg")
    print(f"   RMSE: {rmse:.2f} mg")
    print(f"   MAPE: {mape:.1f}%")
    print(f"   Symmetric MAPE: {smape:.1f}%")
    print(f"   Median AE: {median_ae:.2f} mg")
    print(f"   R²: {r2:.4f}")
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'smape': smape,
        'median_ae': median_ae,
        'r2': r2
    }


def create_all_samples_fast(use_cleaned=True):
    """Create ALL samples that have biomass data (no image loading, no limits)."""
    
    # Use cleaned dataset by default
    if use_cleaned and os.path.exists('bin_results/bin_id_biomass_mapping_cleaned.csv'):
        bin_mapping_csv = 'bin_results/bin_id_biomass_mapping_cleaned.csv'
        print(f" Loading CLEANED data...")
    else:
        bin_mapping_csv = 'bin_results/bin_id_biomass_mapping.csv'
        print(f" Loading original data...")
    
    image_index_json = 'image_index.json'
    
    df = pd.read_csv(bin_mapping_csv)
    
    with open(image_index_json, 'r') as f:
        image_index = json.load(f)
    
    # Create ALL samples (same logic as FixedBiomassDataset but faster)
    samples = []
    
    for _, row in df.iterrows():
        bin_id = row['dna_bin']
        weight = row['mean_weight']
        
        # Get BIOSCAN processids for this BIN
        try:
            if isinstance(row['bioscan_processids'], str):
                bioscan_processids = eval(row['bioscan_processids'])
            else:
                bioscan_processids = row['bioscan_processids']
        except:
            bioscan_processids = []
        
        # Fast lookup in image index (don't check if files exist for speed)
        for processid in bioscan_processids:
            if processid in image_index:
                samples.append({
                    'processid': processid,
                    'bin_id': bin_id,
                    'weight': weight
                })
    
    print(f" Created {len(samples)} total samples with biomass data")
    
    # Extract all weights (NO LIMITING - use everything)
    all_weights = np.array([s['weight'] for s in samples])
    
    return all_weights, samples


def evaluate_baseline_fast(use_cleaned=True):
    """Fast baseline evaluation without image loading."""
    
    print(" Fast Baseline Evaluation: Mean Prediction Analysis")
    print("=" * 60)
    
    # Use cleaned dataset by default
    if use_cleaned and os.path.exists('bin_results/bin_id_biomass_mapping_cleaned.csv'):
        bin_mapping_csv = 'bin_results/bin_id_biomass_mapping_cleaned.csv'
        dataset_type = "CLEANED"
        print(f" Loading CLEANED biomass data from {bin_mapping_csv}...")
    else:
        bin_mapping_csv = 'bin_results/bin_id_biomass_mapping.csv'
        dataset_type = "ORIGINAL"
        print(f" Loading original biomass data from {bin_mapping_csv}...")
    
    df = pd.read_csv(bin_mapping_csv)
    biomass_values = df['mean_weight'].values
    
    print(f" {dataset_type} Dataset Statistics:")
    print(f"   Total BIN groups: {len(df)}")
    print(f"   Biomass mean: {biomass_values.mean():.2f} mg")
    print(f"   Biomass std: {biomass_values.std():.2f} mg")
    print(f"   Biomass median: {np.median(biomass_values):.2f} mg")
    print(f"   Biomass range: [{biomass_values.min():.2f}, {biomass_values.max():.2f}] mg")
    
    # Get ALL samples with biomass data
    all_weights, all_samples = create_all_samples_fast(use_cleaned)
    
    print(f"\n Full Evaluation Set Statistics ({len(all_weights)} samples):")
    print(f"   Mean: {all_weights.mean():.2f} mg")
    print(f"   Std: {all_weights.std():.2f} mg") 
    print(f"   Median: {np.median(all_weights):.2f} mg")
    print(f"   Range: [{all_weights.min():.2f}, {all_weights.max():.2f}] mg")
    
    # Strategy 1: Predict overall dataset mean for every sample
    overall_mean = biomass_values.mean()
    mean_predictions = np.full_like(all_weights, overall_mean)
    
    print(f"\n BASELINE 1: Predict Overall Mean ({overall_mean:.2f} mg)")
    baseline1_metrics = calculate_metrics(mean_predictions, all_weights, "Baseline 1 (Overall Mean)")
    
    # Strategy 2: Predict evaluation set mean for every sample  
    eval_mean = all_weights.mean()
    eval_mean_predictions = np.full_like(all_weights, eval_mean)
    
    print(f"\n BASELINE 2: Predict Evaluation Set Mean ({eval_mean:.2f} mg)")
    baseline2_metrics = calculate_metrics(eval_mean_predictions, all_weights, "Baseline 2 (Eval Mean)")
    
    # Strategy 3: Predict median (more robust to outliers)
    overall_median = np.median(biomass_values)
    median_predictions = np.full_like(all_weights, overall_median)
    
    print(f"\n BASELINE 3: Predict Overall Median ({overall_median:.2f} mg)")
    baseline3_metrics = calculate_metrics(median_predictions, all_weights, "Baseline 3 (Overall Median)")
    
    # Summary comparison
    print(f"\n BASELINE COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<20} {'Overall Mean':<15} {'Test Mean':<15} {'Median':<15}")
    print("-" * 70)
    print(f"{'MAE (mg)':<20} {baseline1_metrics['mae']:<15.2f} {baseline2_metrics['mae']:<15.2f} {baseline3_metrics['mae']:<15.2f}")
    print(f"{'RMSE (mg)':<20} {baseline1_metrics['rmse']:<15.2f} {baseline2_metrics['rmse']:<15.2f} {baseline3_metrics['rmse']:<15.2f}")
    print(f"{'MAPE (%)':<20} {baseline1_metrics['mape']:<15.1f} {baseline2_metrics['mape']:<15.1f} {baseline3_metrics['mape']:<15.1f}")
    print(f"{'SMAPE (%)':<20} {baseline1_metrics['smape']:<15.1f} {baseline2_metrics['smape']:<15.1f} {baseline3_metrics['smape']:<15.1f}")
    print(f"{'R²':<20} {baseline1_metrics['r2']:<15.4f} {baseline2_metrics['r2']:<15.4f} {baseline3_metrics['r2']:<15.4f}")
    
    # Find best baseline
    best_mae = min(baseline1_metrics['mae'], baseline2_metrics['mae'], baseline3_metrics['mae'])
    
    print(f"\n Key Insights:")
    print(f"   - Best baseline MAE: {best_mae:.2f} mg")
    print(f"   - Current model MAE: ~0.73 mg (from previous test)")
    
    if best_mae > 0.73:
        improvement = ((best_mae - 0.73) / best_mae * 100)
        print(f"   -  Model beats baseline by {improvement:.1f}%")
    else:
        print(f"   -   Model needs improvement over baseline")
    
    print(f"\n Recommendation for Full Dataset Training:")
    print(f"   - Target MAE: < {best_mae:.2f} mg (beat baseline)")
    print(f"   - Current model shows promise with only 1000 training samples")
    print(f"   - Full dataset (11,826 samples) should improve performance significantly")
    
    return {
        'baseline_mae': best_mae,
        'total_samples_evaluated': len(all_weights),
        'full_samples_available': len(all_samples)
    }


def main():
    """Main function to run fast baseline evaluation."""
    print(" Starting Fast Baseline Evaluation")
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Check required files
    required_files = ['bin_results/bin_id_biomass_mapping.csv', 'image_index.json']
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f" Required file not found: {file_path}")
            return
    
    print(" Required files found")
    
    # Run baseline evaluation on cleaned data
    results = evaluate_baseline_fast(use_cleaned=True)
    
    print(f"\n Fast baseline evaluation completed!")
    print(f" Ready to train on full dataset to beat {results['baseline_mae']:.2f} mg MAE baseline")


if __name__ == "__main__":
    main()