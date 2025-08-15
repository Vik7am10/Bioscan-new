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
<<<<<<< HEAD
    print("🔄 Downloading pretrained models...")
=======
    print(" Downloading pretrained models...")
>>>>>>> 9266c3c71f078214ff961d8a43734944ed545525
    
    # Set cache directory
    cache_dir = os.path.expanduser("~/.cache/torch/hub/checkpoints/")
    os.makedirs(cache_dir, exist_ok=True)
    
<<<<<<< HEAD
    print(f"📁 Cache directory: {cache_dir}")
    
    try:
        # Download ResNet18 pretrained weights
        print("📥 Downloading ResNet18 pretrained weights...")
        model = models.resnet18(pretrained=True)
        print("✅ ResNet18 downloaded successfully")
=======
    print(f" Cache directory: {cache_dir}")
    
    try:
        # Download ResNet18 pretrained weights
        print(" Downloading ResNet18 pretrained weights...")
        model = models.resnet18(pretrained=True)
        print(" ResNet18 downloaded successfully")
>>>>>>> 9266c3c71f078214ff961d8a43734944ed545525
        
        # Verify the model file exists
        resnet18_path = os.path.join(cache_dir, "resnet18-f37072fd.pth")
        if os.path.exists(resnet18_path):
<<<<<<< HEAD
            print(f"✅ Model cached at: {resnet18_path}")
            print(f"📊 File size: {os.path.getsize(resnet18_path) / 1024 / 1024:.2f} MB")
        else:
            print("⚠️  Model file not found in expected location")
            
    except Exception as e:
        print(f"❌ Error downloading models: {e}")
        return False
    
    print("🎉 All pretrained models downloaded successfully!")
=======
            print(f" Model cached at: {resnet18_path}")
            print(f" File size: {os.path.getsize(resnet18_path) / 1024 / 1024:.2f} MB")
        else:
            print("  Model file not found in expected location")
            
    except Exception as e:
        print(f" Error downloading models: {e}")
        return False
    
    print(" All pretrained models downloaded successfully!")
>>>>>>> 9266c3c71f078214ff961d8a43734944ed545525
    return True

if __name__ == "__main__":
    download_pretrained_models()