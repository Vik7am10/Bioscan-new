#!/usr/bin/env python3
"""
Test the trained biomass model outputs and evaluate performance.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from pathlib import Path
import torchvision.models as models
from PIL import Image

# Import the classes from the training script
from fixed_training_pipeline import FixedBiomassDataset, ImprovedBiomassEstimator


def load_trained_model(model_path: str, device: torch.device):
    """Load the trained model from checkpoint."""
    print(f" Loading model from {model_path}...")
    
    # Fix for PyTorch 2.6+ weights_only default change
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = ImprovedBiomassEstimator(dropout_rate=0.3)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    config = checkpoint['config']
    target_scaler = checkpoint['target_scaler']
    
    print(f" Model loaded successfully")
    print(f" Target scaler: mean={target_scaler['mean']:.2f}, std={target_scaler['std']:.2f}")
    
    return model, config, target_scaler, checkpoint


def denormalize_predictions(predictions, target_scaler):
    """Convert normalized predictions back to original scale."""
    return predictions * target_scaler['std'] + target_scaler['mean']


def evaluate_model(model_path: str):
    """Evaluate the trained model and show predictions."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" Using device: {device}")
    
    if torch.cuda.is_available():
        print(f" GPU Details:")
        print(f"   GPU Count: {torch.cuda.device_count()}")
        print(f"   Current GPU: {torch.cuda.current_device()}")
        print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load model
    model, config, target_scaler, checkpoint = load_trained_model(model_path, device)
    
    # Create test transforms
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create test dataset (using validation split for testing)
    test_dataset = FixedBiomassDataset(
        bin_mapping_csv=config['bin_mapping_csv'],
        image_index_json=config['image_index_json'],
        split='val',
        transform=test_transforms,
        max_samples=100,  # Test on 100 samples
        target_scaler=target_scaler
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2
    )
    
    print(f" Testing on {len(test_dataset)} samples...")
    
    # Collect predictions and targets
    all_predictions = []
    all_targets = []
    all_losses = []
    
    criterion = nn.HuberLoss(delta=1.0)
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(test_loader):
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images).squeeze()
            loss = criterion(outputs, targets)
            
            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_losses.append(loss.item())
            
            if batch_idx == 0:  # Show first batch details
                print(f"\n First batch details:")
                print(f"   Batch size: {images.shape[0]}")
                print(f"   Input shape: {images.shape}")
                print(f"   Output range: [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")
                print(f"   Target range: [{targets.min().item():.4f}, {targets.max().item():.4f}]")
    
    # Convert to numpy arrays
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    
    # Denormalize for interpretability
    pred_denorm = denormalize_predictions(predictions, target_scaler)
    target_denorm = denormalize_predictions(targets, target_scaler)
    
    # Calculate metrics
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(mse)
    
    # Denormalized metrics
    mse_denorm = np.mean((pred_denorm - target_denorm) ** 2)
    mae_denorm = np.mean(np.abs(pred_denorm - target_denorm))
    rmse_denorm = np.sqrt(mse_denorm)
    
    avg_loss = np.mean(all_losses)
    
    print(f"\n Model Performance:")
    print(f"   Average Test Loss: {avg_loss:.4f}")
    print(f"   Normalized MSE: {mse:.4f}")
    print(f"   Normalized MAE: {mae:.4f}")
    print(f"   Normalized RMSE: {rmse:.4f}")
    print(f"\n   Denormalized MSE: {mse_denorm:.2f} mg²")
    print(f"   Denormalized MAE: {mae_denorm:.2f} mg")
    print(f"   Denormalized RMSE: {rmse_denorm:.2f} mg")
    
    # Show prediction examples
    print(f"\n Sample Predictions (denormalized):")
    print(f"{'Index':<6} {'Predicted':<12} {'Actual':<12} {'Error':<12}")
    print("-" * 50)
    
    for i in range(min(20, len(predictions))):
        error = abs(pred_denorm[i] - target_denorm[i])
        print(f"{i:<6} {pred_denorm[i]:<12.2f} {target_denorm[i]:<12.2f} {error:<12.2f}")
    
    # Check for constant predictions
    pred_std = np.std(predictions)
    pred_denorm_std = np.std(pred_denorm)
    
    print(f"\n Prediction Variability:")
    print(f"   Normalized prediction std: {pred_std:.4f}")
    print(f"   Denormalized prediction std: {pred_denorm_std:.2f} mg")
    print(f"   Target std: {np.std(target_denorm):.2f} mg")
    
    if pred_std < 0.01:
        print("  WARNING: Model predictions have very low variability (possible constant predictions)")
    else:
        print(" Model shows good prediction variability")
    
    # Training history
    if 'train_losses' in checkpoint and 'val_losses' in checkpoint:
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        
        print(f"\n Training History:")
        print(f"   Total epochs trained: {len(train_losses)}")
        print(f"   Final train loss: {train_losses[-1]:.4f}")
        print(f"   Final val loss: {val_losses[-1]:.4f}")
        print(f"   Best val loss: {min(val_losses):.4f} (epoch {val_losses.index(min(val_losses))+1})")
    
    return {
        'predictions': predictions,
        'targets': targets,
        'pred_denorm': pred_denorm,
        'target_denorm': target_denorm,
        'metrics': {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'mse_denorm': mse_denorm,
            'mae_denorm': mae_denorm,
            'rmse_denorm': rmse_denorm,
            'avg_loss': avg_loss
        }
    }


def main():
    """Main function to test model outputs."""
    # Find the latest model file
    model_dir = "fixed_model_outputs"
    if not os.path.exists(model_dir):
        print(f" Model directory not found: {model_dir}")
        return
    
    model_files = list(Path(model_dir).glob("*.pth"))
    if not model_files:
        print(f" No model files found in {model_dir}")
        return
    
    # Use the latest model
    latest_model = max(model_files, key=os.path.getmtime)
    print(f" Testing latest model: {latest_model}")
    
    # Evaluate model
    results = evaluate_model(str(latest_model))
    
    print(f"\n Model evaluation completed!")


if __name__ == "__main__":
    main()