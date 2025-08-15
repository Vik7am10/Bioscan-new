#!/usr/bin/env python3
"""
Generate actual vs predicted plots for the trained biomass model.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

# Import from training pipeline
from fixed_training_pipeline import FixedBiomassDataset, create_model

def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics."""
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    
    epsilon = 1e-8
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100
    
    correlation = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R²': r2,
        'Correlation': correlation
    }

def load_model_and_predict(model_path, dataset):
    """Load trained model and make predictions."""
    print(f"Loading model from: {model_path}")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Create data loader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
    
    predictions = []
    targets = []
    
    print(f"Making predictions on {len(dataset)} samples...")
    with torch.no_grad():
        for batch_idx, (images, target_batch) in enumerate(dataloader):
            if batch_idx % 50 == 0:
                print(f"  Processed {batch_idx * 32}/{len(dataset)} samples...")
                
            images = images.to(device)
            outputs = model(images)
            
            predictions.extend(outputs.cpu().numpy().flatten())
            targets.extend(target_batch.numpy().flatten())
    
    return np.array(predictions), np.array(targets)

def plot_actual_vs_predicted(y_true, y_pred, output_path="actual_vs_predicted.png"):
    """Create actual vs predicted plot."""
    plt.figure(figsize=(12, 10))
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)
    
    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Analysis', fontsize=16, fontweight='bold')
    
    # 1. Actual vs Predicted scatter plot
    ax1 = axes[0, 0]
    ax1.scatter(y_true, y_pred, alpha=0.6, s=20)
    
    # Perfect prediction line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax1.set_xlabel('Actual Biomass (mg)')
    ax1.set_ylabel('Predicted Biomass (mg)')
    ax1.set_title('Actual vs Predicted')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Residuals plot
    ax2 = axes[0, 1]
    residuals = y_pred - y_true
    ax2.scatter(y_true, residuals, alpha=0.6, s=20)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('Actual Biomass (mg)')
    ax2.set_ylabel('Residuals (Predicted - Actual)')
    ax2.set_title('Residual Plot')
    ax2.grid(True, alpha=0.3)
    
    # 3. Distribution comparison
    ax3 = axes[1, 0]
    ax3.hist(y_true, bins=50, alpha=0.7, label='Actual', density=True)
    ax3.hist(y_pred, bins=50, alpha=0.7, label='Predicted', density=True)
    ax3.set_xlabel('Biomass (mg)')
    ax3.set_ylabel('Density')
    ax3.set_title('Distribution Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Metrics text
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    metrics_text = f"""
    Performance Metrics:
    
    MAE:  {metrics['MAE']:.4f} mg
    RMSE: {metrics['RMSE']:.4f} mg
    MAPE: {metrics['MAPE']:.2f}%
    R²:   {metrics['R²']:.4f}
    Corr: {metrics['Correlation']:.4f}
    
    Dataset Size: {len(y_true):,} samples
    
    Range:
    Actual: {np.min(y_true):.3f} - {np.max(y_true):.3f} mg
    Predicted: {np.min(y_pred):.3f} - {np.max(y_pred):.3f} mg
    """
    
    ax4.text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to: {output_path}")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Generate actual vs predicted plots')
    parser.add_argument('--model_path', help='Path to trained model (.pth file)')
    parser.add_argument('--bin_mapping_csv', default='bin_results/bin_id_biomass_mapping.csv')
    parser.add_argument('--image_index_json', default='image_index.json') 
    parser.add_argument('--output_plot', default='actual_vs_predicted.png')
    parser.add_argument('--test_split', default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--filter_outliers', action='store_true', help='Apply biological scaling outlier filtering')
    parser.add_argument('--max_biomass', type=float, help='Filter samples above this biomass (mg)')
    
    args = parser.parse_args()
    
    # Auto-find model if not specified
    if not args.model_path:
        import glob
        # Look for model files in likely directories
        model_patterns = [
            'enhanced_model_outputs/*.pth',
            'models_mape/*.pth', 
            'models/*.pth',
            '*.pth'
        ]
        
        for pattern in model_patterns:
            files = glob.glob(pattern)
            if files:
                # Get most recent file
                args.model_path = max(files, key=os.path.getmtime)
                print(f"🔍 Auto-detected model: {args.model_path}")
                break
        
        if not args.model_path:
            print("❌ No model file found! Please specify --model_path")
            return
    
    print("🎯 Generating Actual vs Predicted Plot")
    print("="*50)
    print(f"Model: {args.model_path}")
    print(f"Data split: {args.test_split}")
    print(f"Filter outliers: {args.filter_outliers}")
    if args.max_biomass:
        print(f"Max biomass: {args.max_biomass} mg")
    print("="*50)
    
    # Create dataset
    dataset = FixedBiomassDataset(
        bin_mapping_csv=args.bin_mapping_csv,
        image_index_json=args.image_index_json,
        split=args.test_split,
        filter_outliers=args.filter_outliers,
        max_biomass=args.max_biomass
    )
    
    print(f"📊 Dataset loaded: {len(dataset)} samples")
    
    # Make predictions
    predictions, targets = load_model_and_predict(args.model_path, dataset)
    
    # Generate plot
    metrics = plot_actual_vs_predicted(targets, predictions, args.output_plot)
    
    print("\n📈 Performance Summary:")
    for metric, value in metrics.items():
        if metric in ['MAE', 'RMSE']:
            print(f"  {metric}: {value:.4f} mg")
        elif metric == 'MAPE':
            print(f"  {metric}: {value:.2f}%")
        else:
            print(f"  {metric}: {value:.4f}")

if __name__ == "__main__":
    main()