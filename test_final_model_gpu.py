#!/usr/bin/env python3
"""
Test the final trained model with GPU support.
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
        self.backbone = models.resnet18(pretrained=False)
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        return self.backbone(x)

def main():
    print("🧪 FINAL MODEL EVALUATION (GPU)")
    print("=" * 50)
    
    # Force CUDA check
    print(f"🔍 CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🔍 CUDA device count: {torch.cuda.device_count()}")
        print(f"🔍 Current device: {torch.cuda.current_device()}")
        print(f"🔍 Device name: {torch.cuda.get_device_name(0)}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Find latest MAPE-trained model (best performance)
    mape_models_dir = './models_mape'
    if os.path.exists(mape_models_dir):
        model_files = [f for f in os.listdir(mape_models_dir) if f.startswith('fixed_biomass_model_') and f.endswith('.pth')]
        if model_files:
            latest_model = sorted(model_files)[-1]
            model_path = f"{mape_models_dir}/{latest_model}"
            print(f"📁 Using MAPE-trained model: {latest_model}")
        else:
            print("❌ No MAPE models found")
            return
    else:
        # Fallback to regular models directory
        model_files = [f for f in os.listdir('./models') if f.startswith('fixed_biomass_model_') and f.endswith('.pth')]
        if not model_files:
            print("❌ No trained model found")
            return
        latest_model = sorted(model_files)[-1]
        model_path = f"./models/{latest_model}"
        print(f"📁 Using model: {latest_model}")
    
    # Quick test - just run inference on a few samples
    print("\n🚀 QUICK PERFORMANCE TEST")
    print("=" * 30)
    
    # Load target scaler
    target_scaler = calculate_target_scaler('bin_results/bin_id_biomass_mapping_cleaned.csv')
    
    # Create test transforms
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create TRUE TEST dataset (never seen during training)
    test_dataset = FixedBiomassDataset(
        bin_mapping_csv='bin_results/bin_id_biomass_mapping_cleaned.csv',
        image_index_json='image_index.json',
        split='test',  # Use TRUE holdout test set
        transform=test_transforms,
        max_samples=None,  # Use full test set
        target_scaler=target_scaler,
        use_cleaned=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"📊 Quick test set: {len(test_dataset)} samples")
    
    # Load model
    model = ImprovedBiomassEstimator()
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print("✅ Model loaded and moved to GPU")
    
    # Quick inference
    print("🔍 Running quick inference...")
    start_time = time.time()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(test_loader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            outputs = model(images)
            predictions = outputs.squeeze()
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
            print(f"  Batch {batch_idx+1}/{len(test_loader)} - GPU memory: {torch.cuda.memory_allocated()/1e9:.1f}GB" if device.type == 'cuda' else f"  Batch {batch_idx+1}/{len(test_loader)}")
    
    inference_time = time.time() - start_time
    print(f"⏱️  Inference: {inference_time:.1f}s ({len(test_dataset)/inference_time:.0f} samples/sec)")
    
    # Convert results
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
    
    # Denormalize to real mg values
    def denormalize(normalized_vals, scaler):
        return normalized_vals * scaler['std'] + scaler['mean']
    
    real_predictions = denormalize(all_predictions, target_scaler)
    real_targets = denormalize(all_targets, target_scaler)
    
    # Calculate MAE
    mae = np.mean(np.abs(real_predictions - real_targets))
    baseline_mae = 0.52
    
    print(f"\n🎯 QUICK RESULTS:")
    print(f"   MAE: {mae:.3f} mg")
    print(f"   Baseline: {baseline_mae:.3f} mg")
    
    if mae < baseline_mae:
        improvement = ((baseline_mae - mae) / baseline_mae) * 100
        print(f"   ✅ BEATS BASELINE by {improvement:.1f}%!")
    else:
        print(f"   ❌ Below baseline")
    
    # Sample predictions
    print(f"\n🔍 Sample predictions:")
    print(f"{'Actual':<10} {'Predicted':<10} {'Error':<10}")
    print("-" * 35)
    for i in range(min(5, len(real_targets))):
        actual = real_targets[i]
        predicted = real_predictions[i]
        error = abs(actual - predicted)
        print(f"{actual:<10.3f} {predicted:<10.3f} {error:<10.3f}")
    
    print(f"\n✅ Quick test completed!")
    print(f"Model {'SUCCEEDS' if mae < baseline_mae else 'FAILS'} baseline test")
    
    return mae < baseline_mae

if __name__ == "__main__":
    main()