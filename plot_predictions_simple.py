#!/usr/bin/env python3
"""
Simple plotting script that doesn't depend on the broken training pipeline.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import torchvision.models as models

# Import from training pipeline
try:
    from fixed_training_pipeline import apply_dataset_filters, filter_outliers_scaling_law
    HAVE_FILTERING = True
except ImportError:
    print("Warning: Could not import outlier filtering functions")
    HAVE_FILTERING = False

class SimpleBiomassDataset(Dataset):
    """Simplified dataset class for plotting."""
    
    def __init__(self, bin_mapping_csv, image_index_json, split='test', max_biomass=None, filter_outliers=False):
        # Load image index
        with open(image_index_json, 'r') as f:
            self.image_index = json.load(f)
        print(f"Loaded {len(self.image_index)} images")
        
        # Load BIN mapping
        self.bin_df = pd.read_csv(bin_mapping_csv)
        print(f"Loaded {len(self.bin_df)} BIN mappings")
        
        # Apply dataset filtering if specified
        if (max_biomass is not None or filter_outliers) and HAVE_FILTERING:
            config = {}
            if max_biomass is not None:
                config['max_biomass'] = max_biomass
            if filter_outliers:
                config['filter_outliers'] = True
            
            print("Applying dataset filters...")
            self.bin_df = apply_dataset_filters(self.bin_df, config)
        elif max_biomass is not None:
            # Simple biomass filtering if advanced filtering not available
            original_count = len(self.bin_df)
            self.bin_df = self.bin_df[self.bin_df['mean_weight'] <= max_biomass]
            print(f"Simple filter: {len(self.bin_df)}/{original_count} BINs with biomass ≤ {max_biomass}mg")
        
        # Create samples
        self.samples = []
        found_bins = 0
        
        # Debug: show first few BIN IDs from each source
        print(f"First 5 BIN IDs from CSV: {list(self.bin_df['dna_bin'].head())}")
        print(f"First 5 BIN IDs from image index: {list(self.image_index.keys())[:5]}")
        
        for _, row in self.bin_df.iterrows():
            bin_id = row['dna_bin']
            biomass = row['mean_weight']
            
            # Parse process IDs from the string list
            process_ids_str = row['bioscan_processids']
            if pd.notna(process_ids_str):
                # Convert string representation of list to actual list
                import ast
                try:
                    process_ids = ast.literal_eval(process_ids_str)
                    
                    # Check if any of these process IDs have images
                    for process_id in process_ids:
                        if process_id in self.image_index:
                            found_bins += 1
                            self.samples.append({
                                'image_path': self.image_index[process_id],
                                'biomass': biomass,
                                'bin_id': bin_id,
                                'process_id': process_id
                            })
                            break  # Only take first available image per BIN
                except:
                    continue
        
        print(f"Found {found_bins}/{len(self.bin_df)} matching BIN IDs")
        print(f"Created {len(self.samples)} image-biomass pairs")
        
        # Split data (70/15/15)
        np.random.seed(42)
        np.random.shuffle(self.samples)
        
        n = len(self.samples)
        if split == 'train':
            self.samples = self.samples[:int(0.7 * n)]
        elif split == 'val':
            self.samples = self.samples[int(0.7 * n):int(0.85 * n)]
        elif split == 'test':
            self.samples = self.samples[int(0.85 * n):]
        
        print(f"{split} split: {len(self.samples)} samples")
        
        # Calculate target statistics for normalization
        targets = [s['biomass'] for s in self.samples]
        self.target_mean = np.mean(targets)
        self.target_std = np.std(targets)
        print(f"Target stats - Mean: {self.target_mean:.3f}, Std: {self.target_std:.3f}")
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(sample['image_path']).convert('RGB')
            image = self.transform(image)
        except:
            # Return black image if loading fails
            image = torch.zeros(3, 224, 224)
        
        # Normalize target (will be updated if we load scaler from checkpoint)
        if hasattr(self, 'target_std') and self.target_std > 0:
            target = (sample['biomass'] - self.target_mean) / self.target_std
        else:
            target = sample['biomass']  # Use raw if no normalization available
        
        return image, torch.tensor(target, dtype=torch.float32)

class SimpleBiomassEstimator(nn.Module):
    """Simplified model class matching the saved model."""
    
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        
        # Load pretrained ResNet18
        self.backbone = models.resnet18(pretrained=False)
        
        # Replace final layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        return self.backbone(x)

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

def plot_results(y_true, y_pred, dataset, output_path="predictions_plot.png"):
    """Create comprehensive results plot."""
    
    # Denormalize predictions back to mg
    y_true_mg = y_true * dataset.target_std + dataset.target_mean
    y_pred_mg = y_pred * dataset.target_std + dataset.target_mean
    
    # Debug: print ranges
    print(f"DEBUG - Normalized ranges:")
    print(f"  y_true: {y_true.min():.3f} to {y_true.max():.3f}")
    print(f"  y_pred: {y_pred.min():.3f} to {y_pred.max():.3f}")
    print(f"DEBUG - Denormalized ranges:")
    print(f"  y_true_mg: {y_true_mg.min():.3f} to {y_true_mg.max():.3f}")
    print(f"  y_pred_mg: {y_pred_mg.min():.3f} to {y_pred_mg.max():.3f}")
    print(f"DEBUG - Target stats: mean={dataset.target_mean:.3f}, std={dataset.target_std:.3f}")
    
    # Calculate metrics on real values
    metrics = calculate_metrics(y_true_mg, y_pred_mg)
    
    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Biomass Prediction Results', fontsize=16, fontweight='bold')
    
    # 1. Actual vs Predicted
    ax1 = axes[0, 0]
    ax1.scatter(y_true_mg, y_pred_mg, alpha=0.6, s=20)
    
    # Perfect prediction line
    min_val = min(np.min(y_true_mg), np.min(y_pred_mg))
    max_val = max(np.max(y_true_mg), np.max(y_pred_mg))
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax1.set_xlabel('Actual Biomass (mg)')
    ax1.set_ylabel('Predicted Biomass (mg)')
    ax1.set_title('Actual vs Predicted')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Residuals
    ax2 = axes[0, 1]
    residuals = y_pred_mg - y_true_mg
    ax2.scatter(y_true_mg, residuals, alpha=0.6, s=20)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('Actual Biomass (mg)')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residual Plot')
    ax2.grid(True, alpha=0.3)
    
    # 3. Distributions
    ax3 = axes[1, 0]
    ax3.hist(y_true_mg, bins=50, alpha=0.7, label='Actual', density=True)
    ax3.hist(y_pred_mg, bins=50, alpha=0.7, label='Predicted', density=True)
    ax3.set_xlabel('Biomass (mg)')
    ax3.set_ylabel('Density')
    ax3.set_title('Distribution Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Metrics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    metrics_text = f"""
    Performance Metrics:
    
    MAE:  {metrics['MAE']:.4f} mg
    RMSE: {metrics['RMSE']:.4f} mg
    MAPE: {metrics['MAPE']:.2f}%
    R²:   {metrics['R²']:.4f}
    Corr: {metrics['Correlation']:.4f}
    
    Dataset: {len(y_true):,} samples
    
    Range:
    Actual: {np.min(y_true_mg):.3f} - {np.max(y_true_mg):.3f} mg
    Predicted: {np.min(y_pred_mg):.3f} - {np.max(y_pred_mg):.3f} mg
    """
    
    ax4.text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved: {output_path}")
    
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--bin_mapping_csv', default='bin_results/bin_id_biomass_mapping.csv')
    parser.add_argument('--image_index_json', default='image_index.json')
    parser.add_argument('--output_plot', default='predictions_simple.png')
    parser.add_argument('--max_biomass', type=float, help='Filter samples above this biomass (mg)')
    parser.add_argument('--filter_outliers', action='store_true', help='Apply biological scaling law outlier filtering')
    
    args = parser.parse_args()
    
    print("📈 Simple Biomass Prediction Plot")
    print("=" * 40)
    print(f"Model: {args.model_path}")
    print("=" * 40)
    
    # Create dataset
    dataset = SimpleBiomassDataset(
        args.bin_mapping_csv,
        args.image_index_json,
        split='test',
        max_biomass=args.max_biomass,
        filter_outliers=args.filter_outliers
    )
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleBiomassEstimator(dropout_rate=0.3)
    
    # Load checkpoint (contains metadata + model weights)
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    
    # Extract model state dict and target scaler from checkpoint
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Loaded model from checkpoint")
        
        # Use the same target scaler as training if available
        if 'target_scaler' in checkpoint:
            original_scaler = checkpoint['target_scaler']
            print(f"Original training normalization: mean={original_scaler['mean']:.3f}, std={original_scaler['std']:.3f}")
            
            # Override our dataset's target stats with the training ones
            dataset.target_mean = original_scaler['mean']
            dataset.target_std = original_scaler['std']
            print("✅ Using original training normalization")
    else:
        # Fallback if it's just the state dict
        model.load_state_dict(checkpoint)
        print("✅ Loaded model state dict directly")
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded on {device}")
    
    # Make predictions
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
    
    predictions = []
    targets = []
    
    print("Making predictions...")
    with torch.no_grad():
        for batch_idx, (images, target_batch) in enumerate(dataloader):
            if batch_idx % 20 == 0:
                print(f"  Batch {batch_idx}/{len(dataloader)}")
                
            images = images.to(device)
            outputs = model(images)
            
            predictions.extend(outputs.cpu().numpy().flatten())
            targets.extend(target_batch.numpy().flatten())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    print(f"✅ Generated {len(predictions)} predictions")
    
    # Create plot
    metrics = plot_results(targets, predictions, dataset, args.output_plot)
    
    print("\n📊 Results:")
    for k, v in metrics.items():
        if k in ['MAE', 'RMSE']:
            print(f"  {k}: {v:.4f} mg")
        elif k == 'MAPE':
            print(f"  {k}: {v:.2f}%")
        else:
            print(f"  {k}: {v:.4f}")

if __name__ == "__main__":
    main()