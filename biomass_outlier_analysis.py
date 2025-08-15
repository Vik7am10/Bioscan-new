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


<<<<<<< HEAD
def load_dataset_with_bioscan_features(max_samples=1000):
    """Load dataset using BIOSCAN5M image_measurement_value for correlation analysis."""
    
    # Load BIN mapping data
    bin_mapping_csv = 'bin_results/bin_id_biomass_mapping.csv'
    bioscan_metadata_csv = os.path.expanduser('~/links/scratch/bioscan_data/bioscan5m/metadata/csv/BIOSCAN_5M_Insect_Dataset_metadata.csv')
    
    print(f" Loading biomass data...")
    df_bins = pd.read_csv(bin_mapping_csv)
    
    print(f" Loading BIOSCAN metadata with image measurements...")
    df_bioscan = pd.read_csv(bioscan_metadata_csv)
    
    # Create samples by joining on processid
    samples = []
    
    print(f" Creating samples with BIOSCAN image measurements...")
    for _, bin_row in df_bins.iterrows():
        bin_id = bin_row['dna_bin']
        weight = bin_row['mean_weight']
        specimen_count = bin_row['specimen_count']
        
        # Get BIOSCAN processids for this BIN
        try:
            if isinstance(bin_row['bioscan_processids'], str):
                bioscan_processids = eval(bin_row['bioscan_processids'])
            else:
                bioscan_processids = bin_row['bioscan_processids']
        except:
            bioscan_processids = []
        
        # Find matching BIOSCAN records
        for processid in bioscan_processids:
            bioscan_match = df_bioscan[df_bioscan['processid'] == processid]
            if not bioscan_match.empty:
                bioscan_record = bioscan_match.iloc[0]
                pixel_area = bioscan_record['image_measurement_value']
                if pd.notna(pixel_area) and pixel_area > 0:  # Valid measurement
                    samples.append({
                        'processid': processid,
                        'bin_id': bin_id,
                        'weight': weight,
                        'specimen_count': specimen_count,
                        'pixel_area': pixel_area,
                        'species': bioscan_record.get('species', 'Unknown'),
                        'family': bioscan_record.get('family', 'Unknown'),
                        'order': bioscan_record.get('order', 'Unknown')
                    })
    
    print(f" Created {len(samples)} samples with BIOSCAN measurements")
=======
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
>>>>>>> 9266c3c71f078214ff961d8a43734944ed545525
    
    # Limit for analysis speed
    if max_samples and len(samples) > max_samples:
        samples = random.sample(samples, max_samples)
        print(f" Limited to {max_samples} samples for analysis")
    
    return samples


<<<<<<< HEAD
def create_scaling_law_plot(df_analysis, outlier_samples):
    """Create visualization of biological scaling law with outliers highlighted."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Calculate scaling variables
    sqrt_area = np.sqrt(df_analysis['pixel_area'].values)
    cube_root_mass = np.cbrt(df_analysis['weight'].values)
    
    # Fit scaling model for reference line
    from sklearn.linear_model import HuberRegressor
    scaling_model = HuberRegressor(epsilon=1.35)
    scaling_model.fit(sqrt_area.reshape(-1, 1), cube_root_mass)
    
    # Generate prediction line
    sqrt_area_range = np.linspace(sqrt_area.min(), sqrt_area.max(), 100)
    predicted_cube_root_mass = scaling_model.predict(sqrt_area_range.reshape(-1, 1))
    
    # Plot 1: Scaling Law - sqrt(area) vs cube_root(mass)
    inlier_mask = ~df_analysis['processid'].isin(outlier_samples['processid'])
    
    ax1.scatter(sqrt_area[inlier_mask], cube_root_mass[inlier_mask], 
               alpha=0.6, s=20, color='blue', label='Inliers')
    if len(outlier_samples) > 0:
        outlier_sqrt_area = np.sqrt(outlier_samples['pixel_area'].values)
        outlier_cube_root_mass = np.cbrt(outlier_samples['weight'].values)
        ax1.scatter(outlier_sqrt_area, outlier_cube_root_mass, 
                   alpha=0.8, s=40, color='red', label='Outliers')
    
    ax1.plot(sqrt_area_range, predicted_cube_root_mass, 
             'k--', linewidth=2, label='Scaling Law Fit')
    ax1.set_xlabel('√(Pixel Area)')
    ax1.set_ylabel('∛(Mass) [mg^(1/3)]')
    ax1.set_title('Biological Scaling Law:\n√(Area) ∝ ∛(Mass)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Mass vs Pixel Area (log-log)
    ax2.scatter(df_analysis['pixel_area'].values[inlier_mask], 
               df_analysis['weight'].values[inlier_mask],
               alpha=0.6, s=20, color='blue', label='Inliers')
    if len(outlier_samples) > 0:
        ax2.scatter(outlier_samples['pixel_area'].values, 
                   outlier_samples['weight'].values,
                   alpha=0.8, s=40, color='red', label='Outliers')
    
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Pixel Area')
    ax2.set_ylabel('Mass [mg]')
    ax2.set_title('Mass vs Pixel Area\n(Log-Log Scale)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Residuals from scaling law
    predicted_mass = scaling_model.predict(sqrt_area.reshape(-1, 1))**3
    residuals = df_analysis['weight'].values - predicted_mass
    relative_residuals = residuals / (predicted_mass + 1e-6)
    
    ax3.scatter(predicted_mass[inlier_mask], relative_residuals[inlier_mask],
               alpha=0.6, s=20, color='blue', label='Inliers')
    if len(outlier_samples) > 0:
        outlier_predicted_mass = outlier_samples['predicted_mass'].values
        outlier_relative_error = outlier_samples['relative_error'].values
        # Convert to residual format (can be negative)
        outlier_relative_residuals = outlier_relative_error * np.sign(outlier_samples['weight'].values - outlier_predicted_mass)
        ax3.scatter(outlier_predicted_mass, outlier_relative_residuals,
                   alpha=0.8, s=40, color='red', label='Outliers')
    
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax3.set_xlabel('Predicted Mass [mg]')
    ax3.set_ylabel('Relative Residual')
    ax3.set_title('Scaling Law Residuals')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Distribution of outlier types by mass range
    if len(outlier_samples) > 0:
        mass_bins = np.logspace(np.log10(df_analysis['weight'].min()), 
                               np.log10(df_analysis['weight'].max()), 20)
        
        # Count total samples in each bin
        total_counts, _ = np.histogram(df_analysis['weight'], bins=mass_bins)
        outlier_counts, _ = np.histogram(outlier_samples['weight'], bins=mass_bins)
        
        # Calculate outlier percentage
        outlier_percentage = np.divide(outlier_counts, total_counts, 
                                     out=np.zeros_like(outlier_counts, dtype=float), 
                                     where=total_counts!=0) * 100
        
        bin_centers = (mass_bins[:-1] + mass_bins[1:]) / 2
        
        ax4.bar(bin_centers, outlier_percentage, width=np.diff(mass_bins), 
               alpha=0.7, color='orange', edgecolor='black')
        ax4.set_xscale('log')
        ax4.set_xlabel('Mass [mg]')
        ax4.set_ylabel('Outlier Percentage [%]')
        ax4.set_title('Outlier Rate by Mass Range')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No outliers detected', ha='center', va='center', 
                transform=ax4.transAxes, fontsize=14)
        ax4.set_title('Outlier Rate by Mass Range')
    
    plt.tight_layout()
    plt.savefig('scaling_law_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   Scaling law visualization saved: scaling_law_analysis.png")


=======
>>>>>>> 9266c3c71f078214ff961d8a43734944ed545525
def analyze_biomass_correlations(max_samples=1000):
    """Analyze correlations between image features and biomass."""
    
    print(" Biomass Outlier Analysis: Image Features vs Biomass")
    print("=" * 60)
    
<<<<<<< HEAD
    # Check if we can resume from saved data
    if os.path.exists('biomass_feature_analysis.csv'):
        print(" Found existing analysis data, loading...")
        df_analysis = pd.read_csv('biomass_feature_analysis.csv')
        print(f" Loaded {len(df_analysis)} samples from checkpoint")
    else:
        # Load samples
        samples = load_dataset_with_bioscan_features(max_samples)
        
        # Calculate features for each image
        print(f" Calculating image features for {len(samples)} samples...")
        
        analysis_data = []
        
        for sample in tqdm(samples, desc="Processing samples"):
            # Use BIOSCAN measurements directly - no image processing needed
            pixel_area = sample['pixel_area']
=======
    # Load samples
    samples = load_dataset_with_features(max_samples)
    
    # Calculate features for each image
    print(f" Calculating image features for {len(samples)} samples...")
    
    analysis_data = []
    failed_count = 0
    
    for sample in tqdm(samples, desc="Processing images"):
        features = calculate_image_features(sample['image_path'])
        if features:
>>>>>>> 9266c3c71f078214ff961d8a43734944ed545525
            data_point = {
                'processid': sample['processid'],
                'bin_id': sample['bin_id'],
                'weight': sample['weight'],
                'specimen_count': sample['specimen_count'],
<<<<<<< HEAD
                'pixel_area': pixel_area,
                'cube_area': pixel_area ** 1.5,  # Volume proxy
                'sqrt_area': np.sqrt(pixel_area),  # Linear dimension proxy
                'species': sample['species'],
                'family': sample['family'],
                'order': sample['order']
            }
            analysis_data.append(data_point)
        
        print(f" Successfully processed {len(analysis_data)} samples")
        
        # Convert to DataFrame
        df_analysis = pd.DataFrame(analysis_data)
        
        # Save checkpoint
        print(" Saving checkpoint...")
        df_analysis.to_csv('biomass_feature_analysis.csv', index=False)
        print(" Checkpoint saved!")
    
    # Set threading to avoid BLAS issues
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
=======
                **features
            }
            analysis_data.append(data_point)
        else:
            failed_count += 1
    
    print(f" Successfully processed {len(analysis_data)} samples")
    print(f" Failed to process {failed_count} samples")
    
    # Convert to DataFrame
    df_analysis = pd.DataFrame(analysis_data)
>>>>>>> 9266c3c71f078214ff961d8a43734944ed545525
    
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
    
<<<<<<< HEAD
    feature_cols = ['pixel_area', 'cube_area', 'sqrt_area']
=======
    feature_cols = ['pixel_area', 'cube_area', 'sqrt_area', 'bbox_area', 
                   'contour_area', 'aspect_ratio', 'solidity']
>>>>>>> 9266c3c71f078214ff961d8a43734944ed545525
    
    correlations = {}
    
    for feature in feature_cols:
        if feature in df_analysis.columns:
<<<<<<< HEAD
            try:
                # Use numpy correlation to avoid BLAS threading issues
                r_pearson = np.corrcoef(df_analysis[feature], df_analysis['weight'])[0,1]
                
                # Simple rank correlation
                rank_feature = df_analysis[feature].rank()
                rank_weight = df_analysis['weight'].rank()
                r_spearman = np.corrcoef(rank_feature, rank_weight)[0,1]
                
                correlations[feature] = {
                    'pearson_r': r_pearson,
                    'pearson_p': 0.0,  # Skip p-value calculation to avoid threading issues
                    'spearman_r': r_spearman,
                    'spearman_p': 0.0
                }
                
                print(f"{feature:<15} | Pearson: {r_pearson:>6.3f} | Spearman: {r_spearman:>6.3f}")
            except Exception as e:
                print(f"{feature:<15} | Error calculating correlation: {e}")
                correlations[feature] = {
                    'pearson_r': 0.0,
                    'pearson_p': 1.0,
                    'spearman_r': 0.0,
                    'spearman_p': 1.0
                }
=======
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
>>>>>>> 9266c3c71f078214ff961d8a43734944ed545525
    
    # Find best correlating features
    best_pearson = max(correlations.items(), key=lambda x: abs(x[1]['pearson_r']))
    best_spearman = max(correlations.items(), key=lambda x: abs(x[1]['spearman_r']))
    
    print(f"\n Best Correlations:")
    print(f"   Pearson: {best_pearson[0]} (r={best_pearson[1]['pearson_r']:.3f})")
    print(f"   Spearman: {best_spearman[0]} (r={best_spearman[1]['spearman_r']:.3f})")
    
<<<<<<< HEAD
    # BIOLOGICAL SCALING LAW OUTLIER DETECTION
    print(f"\n Biological Scaling Law Outlier Detection:")
    print("=" * 50)
    print("Theory: sqrt(pixel_area) ∝ cube_root(mass) for isometric scaling")
    
    # Calculate scaling features
    sqrt_area = np.sqrt(df_analysis['pixel_area'].values)
    cube_root_mass = np.cbrt(df_analysis['weight'].values)
    
    print(f"   sqrt(pixel_area) range: [{sqrt_area.min():.1f}, {sqrt_area.max():.1f}]")
    print(f"   cube_root(mass) range: [{cube_root_mass.min():.3f}, {cube_root_mass.max():.3f}]")
    
    # Calculate scaling coefficient for perfect isometric relationship
    # Use robust regression to find the expected scaling line
    from sklearn.linear_model import HuberRegressor
    scaling_model = HuberRegressor(epsilon=1.35)
    scaling_model.fit(sqrt_area.reshape(-1, 1), cube_root_mass)
    
    # Predict expected cube_root_mass from sqrt_area
    expected_cube_root_mass = scaling_model.predict(sqrt_area.reshape(-1, 1))
    scaling_coefficient = scaling_model.coef_[0]
    scaling_intercept = scaling_model.intercept_
    
    print(f"   Scaling relationship: cube_root(mass) = {scaling_coefficient:.6f} * sqrt(area) + {scaling_intercept:.6f}")
    
    # Calculate deviations from scaling law (biological outliers)
    scaling_residuals = cube_root_mass - expected_cube_root_mass
    
    # Convert back to mass space for interpretability
    mass_residuals = (cube_root_mass + scaling_residuals)**3 - df_analysis['weight'].values
    
    # Identify outliers using both absolute and relative thresholds
    # Method 1: Statistical outliers (3-sigma rule)
    abs_threshold = 3 * np.std(scaling_residuals)
    abs_outlier_mask = np.abs(scaling_residuals) > abs_threshold
    
    # Method 2: Relative outliers (mass significantly different from scaling prediction)
    predicted_mass = expected_cube_root_mass**3
    relative_error = np.abs(df_analysis['weight'].values - predicted_mass) / (predicted_mass + 1e-6)  # Avoid division by zero
    rel_threshold = np.percentile(relative_error, 95)  # Top 5% relative errors
    rel_outlier_mask = relative_error > rel_threshold
    
    # Combined outlier detection (either method flags it)
    outlier_mask = abs_outlier_mask | rel_outlier_mask
    outlier_samples = df_analysis[outlier_mask].copy()
    
    # Add scaling analysis columns
    outlier_samples['sqrt_area'] = sqrt_area[outlier_mask]
    outlier_samples['cube_root_mass'] = cube_root_mass[outlier_mask]
    outlier_samples['expected_cube_root_mass'] = expected_cube_root_mass[outlier_mask]
    outlier_samples['scaling_residual'] = scaling_residuals[outlier_mask]
    outlier_samples['predicted_mass'] = predicted_mass[outlier_mask]
    outlier_samples['relative_error'] = relative_error[outlier_mask]
    
    print(f"   Absolute threshold (3-sigma): ±{abs_threshold:.4f} cube_root(mg)")
    print(f"   Relative threshold (95th percentile): {rel_threshold:.1%} relative error")
    print(f"   Outliers detected: {len(outlier_samples)} ({len(outlier_samples)/len(df_analysis)*100:.1f}%)")
    
    if len(outlier_samples) > 0:
        print(f"\n Top Biological Outliers (violate scaling law):")
        outlier_samples_sorted = outlier_samples.sort_values('relative_error', ascending=False)
        
        print(f"{'BIN ID':<12} {'Actual':<8} {'Predicted':<10} {'Rel Error':<10} {'Type':<12}")
        print("-" * 65)
        
        for _, row in outlier_samples_sorted.head(15).iterrows():
            actual_mass = row['weight']
            pred_mass = row['predicted_mass']
            rel_err = row['relative_error']
            
            # Classify outlier type
            if actual_mass > pred_mass * 1.5:
                outlier_type = "Too Heavy"
            elif actual_mass < pred_mass * 0.5:
                outlier_type = "Too Light"
            else:
                outlier_type = "Scaling Dev"
            
            print(f"{row['bin_id']:<12} {actual_mass:<8.1f} {pred_mass:<10.1f} {rel_err:<10.1%} {outlier_type:<12}")
    
    # Additional analysis: Check for systematic deviations
    print(f"\n Scaling Law Validation:")
    correlation_sqrt_cbrt = np.corrcoef(sqrt_area, cube_root_mass)[0,1]
    print(f"   Correlation (sqrt_area, cube_root_mass): {correlation_sqrt_cbrt:.4f}")
    print(f"   Expected for perfect scaling: ~1.0")
    
    if correlation_sqrt_cbrt < 0.7:
        print("   WARNING: Low correlation suggests scaling law may not apply well to this dataset")
    elif correlation_sqrt_cbrt > 0.9:
        print("   EXCELLENT: High correlation confirms biological scaling law")
    else:
        print("   GOOD: Moderate correlation supports scaling law approach")
    
    # Create scaling law visualization
    print(f"\n Creating scaling law visualization...")
    create_scaling_law_plot(df_analysis, outlier_samples)
=======
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
>>>>>>> 9266c3c71f078214ff961d8a43734944ed545525
    
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
<<<<<<< HEAD
    print(f"   - scaling_law_analysis.png (visualization)")
=======
>>>>>>> 9266c3c71f078214ff961d8a43734944ed545525
    
    return df_analysis, outlier_samples, correlations


def main():
    """Main function to run biomass outlier analysis."""
    print(" Starting Biomass Outlier Analysis")
    
    # Set random seed
    random.seed(42)
    np.random.seed(42)
    
    # Check required files
<<<<<<< HEAD
    required_files = ['bin_results/bin_id_biomass_mapping.csv', 
                     os.path.expanduser('~/links/scratch/bioscan_data/bioscan5m/metadata/csv/BIOSCAN_5M_Insect_Dataset_metadata.csv')]
=======
    required_files = ['bin_results/bin_id_biomass_mapping.csv', 'image_index.json']
>>>>>>> 9266c3c71f078214ff961d8a43734944ed545525
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f" Required file not found: {file_path}")
            return
    
    print(" Required files found")
    
<<<<<<< HEAD
    # Run analysis on FULL dataset (all samples)
    df_analysis, outliers, correlations = analyze_biomass_correlations(max_samples=None)
=======
    # Run analysis (limit to 1000 samples for speed)
    df_analysis, outliers, correlations = analyze_biomass_correlations(max_samples=1000)
>>>>>>> 9266c3c71f078214ff961d8a43734944ed545525
    
    print(f"\n Recommendations:")
    print(f"   1. Review samples with weight > {df_analysis['weight'].quantile(0.95):.1f} mg (95th percentile)")
    print(f"   2. Consider removing {len(outliers)} outlier samples")
    print(f"   3. Best feature for biomass prediction: {max(correlations.items(), key=lambda x: abs(x[1]['spearman_r']))[0]}")
    print(f"   4. Expected correlation should be positive (larger specimens = higher biomass)")


if __name__ == "__main__":
    main()