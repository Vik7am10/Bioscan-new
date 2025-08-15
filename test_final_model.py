#!/usr/bin/env python3
"""
Test the final trained model against the baseline and get real MAE performance.
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models
from fixed_training_pipeline import FixedBiomassDataset, calculate_target_scaler

class ImprovedBiomassEstimator(nn.Module):
    """Same model architecture as training."""
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.backbone = models.resnet18(pretrained=False)  # Don't need pretrained for testing
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        return self.backbone(x)

def denormalize_targets(normalized_targets, target_scaler):
    """Convert normalized predictions back to real mg values."""
    mean = target_scaler['mean']
    std = target_scaler['std']
    return normalized_targets * std + mean

def calculate_metrics(predictions, targets):
    """Calculate comprehensive evaluation metrics."""
    # Basic metrics
    mae = np.mean(np.abs(predictions - targets))
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    
    # MAPE (handle division by zero)
    epsilon = 1e-8
    mape = np.mean(np.abs((targets - predictions) / (targets + epsilon))) * 100
    
    # R² coefficient
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # Percentile analysis
    errors = np.abs(predictions - targets)
    error_percentiles = {
        '50th': np.percentile(errors, 50),
        '90th': np.percentile(errors, 90),
        '95th': np.percentile(errors, 95),
        '99th': np.percentile(errors, 99)
    }
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape,
        'r2': r2,
        'error_percentiles': error_percentiles,
        'median_error': np.median(errors),
        'max_error': np.max(errors),
        'min_error': np.min(errors)
    }

def test_model():
    """Test the final trained model."""
    
    print("🧪 FINAL MODEL EVALUATION")
    print("=" * 50)
    
    # Check for model file
    model_files = [f for f in os.listdir('./models') if f.startswith('fixed_biomass_model_') and f.endswith('.pth')]
    if not model_files:
        print("❌ No trained model found in ./models/")
        return
    
    # Use the latest model (best one from training)
    latest_model = sorted(model_files)[-1]
    model_path = f"./models/{latest_model}"
    print(f"📁 Loading model: {latest_model}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Load target scaler (same as training)
    target_scaler = calculate_target_scaler('bin_results/bin_id_biomass_mapping_cleaned.csv')
    print(f"📊 Target normalization: mean={target_scaler['mean']:.2f}, std={target_scaler['std']:.2f}")
    
    # Create test transforms (same as validation)
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create test dataset (use validation split)
    print("📁 Loading test dataset...")
    test_dataset = FixedBiomassDataset(
        bin_mapping_csv='bin_results/bin_id_biomass_mapping_cleaned.csv',
        image_index_json='image_index.json',
        split='val',  # Use validation split as test set
        transform=test_transforms,
        max_samples=None,
        target_scaler=target_scaler,
        use_cleaned=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,  # Larger batch for faster inference
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"📊 Test set: {len(test_dataset)} samples")
    
    # Load model
    print("🔄 Loading trained model...")
    model = ImprovedBiomassEstimator()
    
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            # Checkpoint format: contains model_state_dict, config, etc.
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Model loaded from checkpoint format")
        else:
            # Direct state dict format
            model.load_state_dict(checkpoint)
            print("✅ Model loaded from state dict format")
            
        model.to(device)
        model.eval()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        
        # Try to inspect checkpoint structure
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            print(f"🔍 Checkpoint keys: {list(checkpoint.keys())}")
            if 'model_state_dict' in checkpoint:
                print(f"🔍 Model state dict keys (first 5): {list(checkpoint['model_state_dict'].keys())[:5]}")
        except:
            pass
        return
    
    # Run inference
    print("\n🔍 Running inference...")
    start_time = time.time()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(test_loader):
            images = images.to(device)
            targets = targets.to(device)
            
            # Get predictions
            outputs = model(images)
            predictions = outputs.squeeze()
            
            # Store results
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
            if batch_idx % 10 == 0:
                print(f"  Processed batch {batch_idx}/{len(test_loader)}")
    
    # Combine all results
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
    
    inference_time = time.time() - start_time
    print(f"⏱️  Inference completed in {inference_time:.1f} seconds")
    
    # Denormalize predictions and targets to real mg values
    print("\n🔄 Converting to real biomass values...")
    real_predictions = denormalize_targets(all_predictions, target_scaler)
    real_targets = denormalize_targets(all_targets, target_scaler)
    
    # Calculate metrics
    print("\n📊 PERFORMANCE EVALUATION")
    print("=" * 50)
    
    metrics = calculate_metrics(real_predictions, real_targets)
    
    # Display results
    print(f"🎯 **PRIMARY METRICS:**")
    print(f"   MAE: {metrics['mae']:.3f} mg")
    print(f"   RMSE: {metrics['rmse']:.3f} mg")
    print(f"   MAPE: {metrics['mape']:.1f}%")
    print(f"   R²: {metrics['r2']:.4f}")
    
    print(f"\n📈 **ERROR ANALYSIS:**")
    print(f"   Median error: {metrics['median_error']:.3f} mg")
    print(f"   Max error: {metrics['max_error']:.3f} mg")
    print(f"   Min error: {metrics['min_error']:.3f} mg")
    
    print(f"\n📊 **ERROR PERCENTILES:**")
    for percentile, value in metrics['error_percentiles'].items():
        print(f"   {percentile} percentile: {value:.3f} mg")
    
    # Compare to baseline
    baseline_mae = 0.52  # From baseline evaluation
    print(f"\n🏆 **BASELINE COMPARISON:**")
    print(f"   Baseline MAE: {baseline_mae:.3f} mg")
    print(f"   Model MAE: {metrics['mae']:.3f} mg")
    
    if metrics['mae'] < baseline_mae:
        improvement = ((baseline_mae - metrics['mae']) / baseline_mae) * 100
        print(f"   ✅ SUCCESS! Model beats baseline by {improvement:.1f}%")
        print(f"   🎉 Improvement: {baseline_mae - metrics['mae']:.3f} mg MAE reduction")
    else:
        degradation = ((metrics['mae'] - baseline_mae) / baseline_mae) * 100
        print(f"   ❌ Model underperforms baseline by {degradation:.1f}%")
    
    # Sample predictions
    print(f"\n🔍 **SAMPLE PREDICTIONS:**")
    print(f"{'Actual':<10} {'Predicted':<10} {'Error':<10} {'% Error':<10}")
    print("-" * 45)
    
    for i in range(min(10, len(real_targets))):
        actual = real_targets[i]
        predicted = real_predictions[i]
        error = abs(actual - predicted)
        pct_error = (error / (actual + 1e-8)) * 100
        
        print(f"{actual:<10.3f} {predicted:<10.3f} {error:<10.3f} {pct_error:<10.1f}%")
    
    # Data distribution analysis
    print(f"\n📊 **DATA DISTRIBUTION:**")
    print(f"   Target range: [{real_targets.min():.3f}, {real_targets.max():.3f}] mg")
    print(f"   Target mean: {real_targets.mean():.3f} mg")
    print(f"   Target median: {np.median(real_targets):.3f} mg")
    print(f"   Prediction range: [{real_predictions.min():.3f}, {real_predictions.max():.3f}] mg")
    print(f"   Prediction mean: {real_predictions.mean():.3f} mg")
    
    print(f"\n✅ MODEL EVALUATION COMPLETED!")
    print(f"Final MAE: {metrics['mae']:.3f} mg ({'BEATS' if metrics['mae'] < baseline_mae else 'BELOW'} baseline)")
    
    return metrics

if __name__ == "__main__":
    test_model()