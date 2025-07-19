#!/usr/bin/env python3
"""
Optimized ML Pipeline - Uses pre-built image index for fast lookups.
"""

import os
import sys
import json
import argparse
import random
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import torchvision.models as models

class OptimizedBiomassDataset(Dataset):
    """
    Optimized dataset using pre-built image index for fast file lookups.
    """
    
    def __init__(self, 
                 bin_mapping_csv: str,
                 image_index_json: str,
                 split: str = "train",
                 transform: Optional[transforms.Compose] = None,
                 max_samples: int = None):
        """
        Initialize optimized dataset.
        
        Args:
            bin_mapping_csv: Path to BIN mapping results CSV
            image_index_json: Path to pre-built image index JSON
            split: Dataset split ('train', 'val', 'test')
            transform: Image transformations
            max_samples: Maximum number of samples to use
        """
        self.bin_mapping_csv = bin_mapping_csv
        self.image_index_json = image_index_json
        self.split = split
        self.transform = transform
        self.max_samples = max_samples
        
        # Load image index
        print(f"🔄 Loading image index from {image_index_json}...")
        with open(image_index_json, 'r') as f:
            self.image_index = json.load(f)
        print(f"✅ Loaded index with {len(self.image_index)} images")
        
        # Load BIN mapping data
        print(f"🔄 Loading BIN mapping data from {bin_mapping_csv}...")
        self.bin_df = pd.read_csv(bin_mapping_csv)
        print(f"✅ Loaded {len(self.bin_df)} BIN mappings")
        
        # Create sample pairs quickly
        self.samples = self._create_samples_fast()
        
        # Split data
        self.split_samples = self._split_data()
        
        print(f"✅ Optimized dataset initialized:")
        print(f"   - Total BIN mappings: {len(self.bin_df)}")
        print(f"   - Total image samples: {len(self.samples)}")
        print(f"   - {split} split samples: {len(self.split_samples)}")
        
    def _create_samples_fast(self) -> List[Dict]:
        """Create samples using fast image index lookups."""
        samples = []
        
        print("🔄 Creating samples with fast lookups...")
        
        for _, row in self.bin_df.iterrows():
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
            
            # Fast lookup in image index
            for processid in bioscan_processids:
                if processid in self.image_index:
                    image_path = self.image_index[processid]
                    if os.path.exists(image_path):
                        sample = {
                            'image_path': image_path,
                            'weight': weight,
                            'bin_id': bin_id,
                            'processid': processid,
                            'specimen_count': specimen_count
                        }
                        samples.append(sample)
        
        print(f"✅ Created {len(samples)} valid samples")
        
        if self.max_samples and len(samples) > self.max_samples:
            samples = random.sample(samples, self.max_samples)
            print(f"🔄 Limited to {len(samples)} samples")
        else:
            print(f"🔄 Using full dataset: {len(samples)} samples")
        
        return samples
    
    def _split_data(self) -> List[Dict]:
        """Split data into train/val/test."""
        random.shuffle(self.samples)
        
        if self.split == "train":
            return self.samples[:int(0.8 * len(self.samples))]
        elif self.split == "val":
            start_idx = int(0.8 * len(self.samples))
            end_idx = int(0.9 * len(self.samples))
            return self.samples[start_idx:end_idx]
        else:  # test
            return self.samples[int(0.9 * len(self.samples)):]
    
    def __len__(self) -> int:
        return len(self.split_samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset."""
        try:
            sample = self.split_samples[idx]
            
            # Load image
            image = Image.open(sample['image_path']).convert('RGB')
            
            # Apply transformations
            if self.transform:
                image = self.transform(image)
            
            # Get weight
            weight = torch.tensor(sample['weight'], dtype=torch.float32)
            
            return image, weight
                
        except Exception as e:
            print(f"❌ Error loading sample {idx}: {e}")
            # Return zero tensors as fallback
            image = torch.zeros(3, 224, 224)
            weight = torch.tensor(0.0, dtype=torch.float32)
            return image, weight


class SimpleBiomassEstimator(nn.Module):
    """Simple CNN for biomass estimation."""
    def __init__(self):
        super().__init__()
        # Use ResNet18 as backbone
        self.backbone = models.resnet18(pretrained=False)
        self.backbone.fc = nn.Linear(512, 1)
        
    def forward(self, x):
        return self.backbone(x)


def train_model(config: Dict):
    """Train the biomass estimation model."""
    print("🚀 Starting optimized biomass training...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ Using device: {device}")
    
    # Set random seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create transforms
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = OptimizedBiomassDataset(
        bin_mapping_csv=config['bin_mapping_csv'],
        image_index_json=config['image_index_json'],
        split="train",
        transform=train_transforms,
        max_samples=config.get('max_samples', None)
    )
    
    val_dataset = OptimizedBiomassDataset(
        bin_mapping_csv=config['bin_mapping_csv'],
        image_index_json=config['image_index_json'],
        split="val",
        transform=val_transforms,
        max_samples=config.get('max_samples', None)
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    print(f"✅ Created dataloaders: {len(train_dataset)} train, {len(val_dataset)} val")
    
    # Create model
    model = SimpleBiomassEstimator().to(device)
    print(f"✅ Created model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.L1Loss()
    
    # Training loop
    for epoch in range(config['num_epochs']):
        print(f"\n🔄 Epoch {epoch + 1}/{config['num_epochs']}")
        
        # Train
        model.train()
        train_loss = 0.0
        for batch_idx, (images, weights) in enumerate(train_loader):
            images, weights = images.to(device), weights.to(device)
            
            optimizer.zero_grad()
            predictions = model(images).squeeze()
            loss = criterion(predictions, weights)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, weights in val_loader:
                images, weights = images.to(device), weights.to(device)
                predictions = model(images).squeeze()
                loss = criterion(predictions, weights)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"  Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    print("\n✅ Training completed!")
    
    # Save model
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = output_dir / f"optimized_biomass_model_{timestamp}.pth"
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config
    }, model_path)
    
    print(f"💾 Model saved to: {model_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Optimized Biomass Training')
    parser.add_argument('--bin_mapping_csv', required=True, help='BIN mapping CSV path')
    parser.add_argument('--image_index_json', required=True, help='Image index JSON path')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--output_dir', default='./optimized_outputs', help='Output directory')
    parser.add_argument('--max_samples', type=int, help='Max samples for testing')
    
    args = parser.parse_args()
    
    config = {
        'bin_mapping_csv': args.bin_mapping_csv,
        'image_index_json': args.image_index_json,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'output_dir': args.output_dir,
        'max_samples': args.max_samples
    }
    
    print("🎯 Optimized Biomass Training Pipeline")
    print("=" * 50)
    for key, value in config.items():
        print(f"{key}: {value}")
    print("=" * 50)
    
    train_model(config)


if __name__ == "__main__":
    main()