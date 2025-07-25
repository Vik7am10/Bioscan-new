#!/usr/bin/env python3
"""
Download pretrained models on login node where network access is available.
Run this before submitting SLURM jobs.
"""

import torch
import torchvision.models as models
import os

def download_pretrained_models():
    """Download pretrained models to cache directory."""
    print("🔄 Downloading pretrained models...")
    
    # Set cache directory
    cache_dir = os.path.expanduser("~/.cache/torch/hub/checkpoints/")
    os.makedirs(cache_dir, exist_ok=True)
    
    print(f"📁 Cache directory: {cache_dir}")
    
    try:
        # Download ResNet18 pretrained weights
        print("📥 Downloading ResNet18 pretrained weights...")
        model = models.resnet18(pretrained=True)
        print("✅ ResNet18 downloaded successfully")
        
        # Verify the model file exists
        resnet18_path = os.path.join(cache_dir, "resnet18-f37072fd.pth")
        if os.path.exists(resnet18_path):
            print(f"✅ Model cached at: {resnet18_path}")
            print(f"📊 File size: {os.path.getsize(resnet18_path) / 1024 / 1024:.2f} MB")
        else:
            print("⚠️  Model file not found in expected location")
            
    except Exception as e:
        print(f"❌ Error downloading models: {e}")
        return False
    
    print("🎉 All pretrained models downloaded successfully!")
    return True

if __name__ == "__main__":
    download_pretrained_models()