#!/usr/bin/env python3
"""
Analyze correlation between image features (pixel area, cube area) and biomass
to identify potential outliers in the dataset.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from pathlib import Path
import random
from tqdm import tqdm


def calculate_image_features(image_path):
    """
    Calculate image features: pixel area and volume proxy.
    Returns area, cube_area, and other morphological features.
    """
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Create binary mask (remove background)
        # Assume specimens are darker than background
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Calculate pixel area (number of foreground pixels)
        pixel_area = np.sum(binary > 0)
        
        # Calculate bounding box area
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Get largest contour (main specimen)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            bbox_area = w * h
            
            # Calculate contour area
            contour_area = cv2.contourArea(largest_contour)
            
            # Calculate aspect ratio
            aspect_ratio = w / h if h > 0 else 0
            
            # Calculate solidity (area/convex_hull_area)
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            solidity = contour_area / hull_area if hull_area > 0 else 0
            
        else:
            bbox_area = 0
            contour_area = 0
            aspect_ratio = 0
            solidity = 0
        
        # Volume proxies
        cube_area = pixel_area ** 1.5  # Assuming 3D volume scales as area^1.5
        sqrt_area = np.sqrt(pixel_area)  # Linear dimension proxy
        
        return {
            'pixel_area': pixel_area,
            'cube_area': cube_area,
            'sqrt_area': sqrt_area,
            'bbox_area': bbox_area,
            'contour_area': contour_area,
            'aspect_ratio': aspect_ratio,
            'solidity': solidity,
            'image_width': image.shape[1],
            'image_height': image.shape[0]
        }
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def load_dataset_with_features(max_samples=1000):
    """Load dataset and calculate image features for correlation analysis."""
    
    # Load data
    bin_mapping_csv = 'bin_results/bin_id_biomass_mapping.csv'
    image_index_json = 'image_index.json'
    
    print(f" Loading data...")
    df = pd.read_csv(bin_mapping_csv)
    
    with open(image_index_json, 'r') as f:
        image_index = json.load(f)
    
    # Create samples
    samples = []
    
    print(f" Creating samples...")
    for _, row in df.iterrows():
        bin_id = row['dna_bin']
        weight = row['mean_weight']
        specimen_count = row['specimen_count']
        
        # Get BIOSCAN processids for this BIN
        try:
            if isinstance(row['bioscan_processids'], str):
                bioscan_processids = eval(row['bioscan_processids'])
            else:
                bioscan_processids = row['bioscan_processids']
        except:
            bioscan_processids = []
        
        # Add samples for this BIN
        for processid in bioscan_processids:
            if processid in image_index:
                image_path = image_index[processid]
                if os.path.exists(image_path):
                    samples.append({
                        'image_path': image_path,
                        'processid': processid,
                        'bin_id': bin_id,
                        'weight': weight,
                        'specimen_count': specimen_count
                    })
    
    print(f" Created {len(samples)} samples")
    
    # Limit for analysis speed
    if max_samples and len(samples) > max_samples:
        samples = random.sample(samples, max_samples)
        print(f" Limited to {max_samples} samples for analysis")
    
    return samples


def analyze_biomass_correlations(max_samples=1000):
    """Analyze correlations between image features and biomass."""
    
    print(" Biomass Outlier Analysis: Image Features vs Biomass")
    print("=" * 60)
    
    # Load samples
    samples = load_dataset_with_features(max_samples)
    
    # Calculate features for each image
    print(f" Calculating image features for {len(samples)} samples...")
    
    analysis_data = []
    failed_count = 0
    
    for sample in tqdm(samples, desc="Processing images"):
        features = calculate_image_features(sample['image_path'])
        if features:
            data_point = {
                'processid': sample['processid'],
                'bin_id': sample['bin_id'],
                'weight': sample['weight'],
                'specimen_count': sample['specimen_count'],
                **features
            }
            analysis_data.append(data_point)
        else:
            failed_count += 1
    
    print(f" Successfully processed {len(analysis_data)} samples")
    print(f" Failed to process {failed_count} samples")
    
    # Convert to DataFrame
    df_analysis = pd.DataFrame(analysis_data)
    
    # Basic statistics
    print(f"\n Dataset Statistics:")
    print(f"   Total samples: {len(df_analysis)}")
    print(f"   Biomass range: [{df_analysis['weight'].min():.2f}, {df_analysis['weight'].max():.2f}] mg")
    print(f"   Biomass mean: {df_analysis['weight'].mean():.2f} mg")
    print(f"   Biomass median: {df_analysis['weight'].median():.2f} mg")
    print(f"   Biomass std: {df_analysis['weight'].std():.2f} mg")
    
    # Identify potential outliers
    weight_q99 = df_analysis['weight'].quantile(0.99)
    weight_q95 = df_analysis['weight'].quantile(0.95)
    weight_q90 = df_analysis['weight'].quantile(0.90)
    
    print(f"\n Biomass Percentiles:")
    print(f"   90th percentile: {weight_q90:.2f} mg")
    print(f"   95th percentile: {weight_q95:.2f} mg") 
    print(f"   99th percentile: {weight_q99:.2f} mg")
    
    outliers_99 = df_analysis[df_analysis['weight'] > weight_q99]
    outliers_95 = df_analysis[df_analysis['weight'] > weight_q95]
    
    print(f"\n Potential Outliers:")
    print(f"   Above 95th percentile: {len(outliers_95)} samples ({len(outliers_95)/len(df_analysis)*100:.1f}%)")
    print(f"   Above 99th percentile: {len(outliers_99)} samples ({len(outliers_99)/len(df_analysis)*100:.1f}%)")
    
    # Correlation analysis
    print(f"\n Correlation Analysis:")
    print("=" * 50)
    
    feature_cols = ['pixel_area', 'cube_area', 'sqrt_area', 'bbox_area', 
                   'contour_area', 'aspect_ratio', 'solidity']
    
    correlations = {}
    
    for feature in feature_cols:
        if feature in df_analysis.columns:
            # Pearson correlation
            r_pearson, p_pearson = pearsonr(df_analysis[feature], df_analysis['weight'])
            
            # Spearman correlation (rank-based, more robust to outliers)
            r_spearman, p_spearman = spearmanr(df_analysis[feature], df_analysis['weight'])
            
            correlations[feature] = {
                'pearson_r': r_pearson,
                'pearson_p': p_pearson,
                'spearman_r': r_spearman,
                'spearman_p': p_spearman
            }
            
            print(f"{feature:<15} | Pearson: {r_pearson:>6.3f} (p={p_pearson:.3f}) | Spearman: {r_spearman:>6.3f} (p={p_spearman:.3f})")
    
    # Find best correlating features
    best_pearson = max(correlations.items(), key=lambda x: abs(x[1]['pearson_r']))
    best_spearman = max(correlations.items(), key=lambda x: abs(x[1]['spearman_r']))
    
    print(f"\n Best Correlations:")
    print(f"   Pearson: {best_pearson[0]} (r={best_pearson[1]['pearson_r']:.3f})")
    print(f"   Spearman: {best_spearman[0]} (r={best_spearman[1]['spearman_r']:.3f})")
    
    # Outlier detection using best feature
    best_feature = best_spearman[0]  # Use Spearman as it's more robust
    
    print(f"\n Outlier Detection using {best_feature}:")
    
    # Calculate residuals from expected relationship
    x = df_analysis[best_feature].values
    y = df_analysis['weight'].values
    
    # Fit robust regression (resistant to outliers)
    from sklearn.linear_model import HuberRegressor
    huber = HuberRegressor(epsilon=1.35)  # Default epsilon
    huber.fit(x.reshape(-1, 1), y)
    
    y_pred = huber.predict(x.reshape(-1, 1))
    residuals = y - y_pred
    
    # Identify outliers using residuals
    residual_threshold = 3 * np.std(residuals)  # 3-sigma rule
    outlier_mask = np.abs(residuals) > residual_threshold
    
    outlier_samples = df_analysis[outlier_mask]
    
    print(f"   Residual threshold: ±{residual_threshold:.2f} mg")
    print(f"   Outliers detected: {len(outlier_samples)} ({len(outlier_samples)/len(df_analysis)*100:.1f}%)")
    
    if len(outlier_samples) > 0:
        print(f"\n Top Outliers by Residual:")
        outlier_samples_sorted = outlier_samples.assign(residual=residuals[outlier_mask])
        outlier_samples_sorted = outlier_samples_sorted.sort_values('residual', key=abs, ascending=False)
        
        print(f"{'BIN ID':<12} {'Weight':<8} {'Feature':<10} {'Predicted':<10} {'Residual':<10}")
        print("-" * 60)
        
        for _, row in outlier_samples_sorted.head(10).iterrows():
            idx = df_analysis.index.get_loc(row.name)
            pred_weight = y_pred[idx]
            residual = residuals[idx]
            feature_val = row[best_feature]
            
            print(f"{row['bin_id']:<12} {row['weight']:<8.1f} {feature_val:<10.0f} {pred_weight:<10.1f} {residual:<10.1f}")
    
    # Save results
    print(f"\n Saving analysis results...")
    
    # Save full analysis
    df_analysis.to_csv('biomass_feature_analysis.csv', index=False)
    
    # Save outliers
    if len(outlier_samples) > 0:
        outlier_samples.to_csv('biomass_outliers.csv', index=False)
    
    # Save correlation results
    corr_df = pd.DataFrame(correlations).T
    corr_df.to_csv('biomass_correlations.csv')
    
    print(f" Analysis completed!")
    print(f" Files saved:")
    print(f"   - biomass_feature_analysis.csv (full analysis)")
    print(f"   - biomass_outliers.csv (outlier samples)")
    print(f"   - biomass_correlations.csv (correlation results)")
    
    return df_analysis, outlier_samples, correlations


def main():
    """Main function to run biomass outlier analysis."""
    print(" Starting Biomass Outlier Analysis")
    
    # Set random seed
    random.seed(42)
    np.random.seed(42)
    
    # Check required files
    required_files = ['bin_results/bin_id_biomass_mapping.csv', 'image_index.json']
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f" Required file not found: {file_path}")
            return
    
    print(" Required files found")
    
    # Run analysis (limit to 1000 samples for speed)
    df_analysis, outliers, correlations = analyze_biomass_correlations(max_samples=1000)
    
    print(f"\n Recommendations:")
    print(f"   1. Review samples with weight > {df_analysis['weight'].quantile(0.95):.1f} mg (95th percentile)")
    print(f"   2. Consider removing {len(outliers)} outlier samples")
    print(f"   3. Best feature for biomass prediction: {max(correlations.items(), key=lambda x: abs(x[1]['spearman_r']))[0]}")
    print(f"   4. Expected correlation should be positive (larger specimens = higher biomass)")


if __name__ == "__main__":
    main()