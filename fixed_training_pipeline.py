#!/usr/bin/env python3
"""
Fixed ML Pipeline - Addresses the constant prediction issue with proper normalization and learning rate.
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

class FixedBiomassDataset(Dataset):
    """
    Fixed dataset with proper target normalization.
    """
    
    def __init__(self, 
                 bin_mapping_csv: str,
                 image_index_json: str,
                 split: str = "train",
                 transform: Optional[transforms.Compose] = None,
                 max_samples: int = None,
                 target_scaler: Optional[Dict] = None):
        """
        Initialize dataset with target normalization.
        
        Args:
            bin_mapping_csv: Path to BIN mapping results CSV
            image_index_json: Path to pre-built image index JSON
            split: Dataset split ('train', 'val', 'test')
            transform: Image transformations
            max_samples: Maximum number of samples to use
            target_scaler: Dict with 'mean' and 'std' for target normalization
        """
        self.bin_mapping_csv = bin_mapping_csv
        self.image_index_json = image_index_json
        self.split = split
        self.transform = transform
        self.max_samples = max_samples
        self.target_scaler = target_scaler
        
        # Load image index
        print(f" Loading image index from {image_index_json}...")
        with open(image_index_json, 'r') as f:
            self.image_index = json.load(f)
        print(f" Loaded index with {len(self.image_index)} images")
        
        # Load BIN mapping data
        print(f" Loading BIN mapping data from {bin_mapping_csv}...")
        self.bin_df = pd.read_csv(bin_mapping_csv)
        print(f" Loaded {len(self.bin_df)} BIN mappings")
        
        # Print biomass statistics
        weights = self.bin_df['mean_weight'].values
        print(f" Biomass statistics:")
        print(f"   - Mean: {weights.mean():.2f} mg")
        print(f"   - Std: {weights.std():.2f} mg")
        print(f"   - Min: {weights.min():.2f} mg")
        print(f"   - Max: {weights.max():.2f} mg")
        print(f"   - Median: {np.median(weights):.2f} mg")
        
        # Create sample pairs quickly
        self.samples = self._create_samples_fast()
        
        # Split data
        self.split_samples = self._split_data()
        
        print(f" Fixed dataset initialized:")
        print(f"   - Total BIN mappings: {len(self.bin_df)}")
        print(f"   - Total image samples: {len(self.samples)}")
        print(f"   - {split} split samples: {len(self.split_samples)}")
        
    def _create_samples_fast(self) -> List[Dict]:
        """Create samples using fast image index lookups."""
        samples = []
        
        print(" Creating samples with fast lookups...")
        
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
                            'processid': processid,
                            'bin_id': bin_id,
                            'weight': weight,
                            'specimen_count': specimen_count
                        }
                        samples.append(sample)
        
        print(f" Created {len(samples)} valid samples")
        
        # Limit samples if specified
        if self.max_samples and len(samples) > self.max_samples:
            samples = random.sample(samples, self.max_samples)
            print(f" Limited to {self.max_samples} samples")
        
        return samples
    
    def _split_data(self) -> List[Dict]:
        """Split data into train/val/test."""
        random.shuffle(self.samples)
        
        if self.split == "train":
            return self.samples[:int(0.8 * len(self.samples))]
        elif self.split == "val":
            return self.samples[int(0.8 * len(self.samples)):]
        else:  # test
            return self.samples[int(0.9 * len(self.samples)):]
    
    def __len__(self):
        return len(self.split_samples)
    
    def __getitem__(self, idx):
        """Get a sample with proper target normalization."""
        try:
            sample = self.split_samples[idx]
            
            # Load image
            image = Image.open(sample['image_path']).convert('RGB')
            if self.transform:
                image = self.transform(image)
            
            # Get weight and normalize if scaler provided
            weight = sample['weight']
            if self.target_scaler:
                weight = (weight - self.target_scaler['mean']) / self.target_scaler['std']
            
            weight = torch.tensor(weight, dtype=torch.float32)
            
            return image, weight
                
        except Exception as e:
            print(f" Error loading sample {idx}: {e}")
            # Return zero tensors as fallback
            image = torch.zeros(3, 224, 224)
            weight = torch.tensor(0.0, dtype=torch.float32)
            return image, weight


class ImprovedBiomassEstimator(nn.Module):
    """Improved CNN for biomass estimation with dropout and better initialization."""
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        # Use ResNet18 as backbone
        self.backbone = models.resnet18(pretrained=True)  # Use pretrained weights
        
        # Replace final layer with improved head
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1)
        )
        
        # Initialize final layer with small weights
        nn.init.normal_(self.backbone.fc[-1].weight, 0, 0.01)
        nn.init.constant_(self.backbone.fc[-1].bias, 0)
        
    def forward(self, x):
        return self.backbone(x)


def calculate_target_scaler(bin_mapping_csv: str) -> Dict:
    """Calculate mean and std for target normalization."""
    df = pd.read_csv(bin_mapping_csv)
    weights = df['mean_weight'].values
    
    scaler = {
        'mean': weights.mean(),
        'std': weights.std()
    }
    
    print(f" Target scaler calculated:")
    print(f"   - Mean: {scaler['mean']:.2f} mg")
    print(f"   - Std: {scaler['std']:.2f} mg")
    
    return scaler


def train_model(config: Dict):
    """Train the improved biomass estimation model."""
    print(" Starting FIXED biomass training...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" Using device: {device}")
    
    # Set random seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Calculate target scaler
    target_scaler = calculate_target_scaler(config['bin_mapping_csv'])
    
    # Create transforms
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = FixedBiomassDataset(
        bin_mapping_csv=config['bin_mapping_csv'],
        image_index_json=config['image_index_json'],
        split='train',
        transform=train_transforms,
        max_samples=config.get('max_samples'),
        target_scaler=target_scaler
    )
    
    val_dataset = FixedBiomassDataset(
        bin_mapping_csv=config['bin_mapping_csv'],
        image_index_json=config['image_index_json'],
        split='val',
        transform=val_transforms,
        max_samples=config.get('max_samples'),
        target_scaler=target_scaler
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f" Created dataloaders: {len(train_dataset)} train, {len(val_dataset)} val")
    
    # Create model
    model = ImprovedBiomassEstimator(dropout_rate=0.3)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f" Created model with {total_params:,} parameters ({trainable_params:,} trainable)")
    
    # Use Huber loss (more robust to outliers)
    criterion = nn.HuberLoss(delta=1.0)
    
    # Use lower learning rate with scheduler
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    patience_limit = 15
    
    train_losses = []
    val_losses = []
    
    for epoch in range(config['num_epochs']):
        print(f"\n Epoch {epoch+1}/{config['num_epochs']}")
        
        # Training
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}, Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                
                outputs = model(images).squeeze()
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        val_losses.append(avg_val_loss)
        
        print(f"  Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save best model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = f"{config['output_dir']}/fixed_biomass_model_{timestamp}.pth"
            os.makedirs(config['output_dir'], exist_ok=True)
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'target_scaler': target_scaler,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'epoch': epoch
            }, model_path)
            
            print(f"   Best model saved: {model_path}")
            
        else:
            patience_counter += 1
            
        if patience_counter >= patience_limit:
            print(f"   Early stopping after {patience_limit} epochs without improvement")
            break
    
    print(" Training completed!")
    print(f" Best model saved to: {model_path}")
    
    return model_path


def main():
    parser = argparse.ArgumentParser(description='Fixed Biomass Training Pipeline')
    parser.add_argument('--bin_mapping_csv', default='bin_results/bin_id_biomass_mapping.csv',
                       help='Path to BIN mapping CSV file')
    parser.add_argument('--image_index_json', default='image_index.json',
                       help='Path to image index JSON file')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                       help='Learning rate (reduced from 0.001)')
    parser.add_argument('--output_dir', default='fixed_model_outputs',
                       help='Output directory for model and results')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to use (None for all - FULL DATASET)')
    
    args = parser.parse_args()
    
    # Convert args to config dict
    config = vars(args)
    
    # Print configuration
    print(" Fixed Biomass Training Configuration:")
    print("==================================================")
    for key, value in config.items():
        print(f"{key}: {value}")
    print("==================================================")
    
    # Check required files
    required_files = [args.bin_mapping_csv, args.image_index_json]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f" Required file not found: {file_path}")
            sys.exit(1)
    
    print(" Required files found. Starting fixed training...")
    
    # Train model
    model_path = train_model(config)
    
    print(f"\n Fixed training completed!")
    print(f" Model saved to: {model_path}")

if __name__ == "__main__":
    main()