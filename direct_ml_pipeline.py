#!/usr/bin/env python3
"""
Direct ML Pipeline - Using BIN mapping results and BIOSCAN images
Bypasses bioscan-dataset package, uses direct file access for images.
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
from glob import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import torchvision.models as models

class DirectBiomassDataset(Dataset):
    """
    Direct dataset using BIN mapping results and BIOSCAN image files.
    """
    
    def __init__(self, 
                 bin_mapping_csv: str,
                 bioscan_images_root: str,
                 split: str = "train",
                 transform: Optional[transforms.Compose] = None,
                 use_scale_features: bool = False,
                 max_samples: int = None):
        """
        Initialize the direct dataset.
        
        Args:
            bin_mapping_csv: Path to BIN mapping results CSV
            bioscan_images_root: Root directory for BIOSCAN images  
            split: Dataset split ('train', 'val', 'test')
            transform: Image transformations
            use_scale_features: Whether to include scale features
            max_samples: Maximum number of samples to use (for testing)
        """
        self.bin_mapping_csv = bin_mapping_csv
        self.bioscan_images_root = bioscan_images_root
        self.split = split
        self.transform = transform
        self.use_scale_features = use_scale_features
        self.max_samples = max_samples
        
        # Load BIN mapping data
        print(f"🔄 Loading BIN mapping data from {bin_mapping_csv}...")
        self.bin_df = pd.read_csv(bin_mapping_csv)
        print(f"✅ Loaded {len(self.bin_df)} BIN mappings")
        
        # Create sample pairs (BIN -> weight, image_paths)
        self.samples = self._create_samples()
        
        # Split data
        self.split_samples = self._split_data()
        
        print(f"✅ Direct dataset initialized:")
        print(f"   - Total BIN mappings: {len(self.bin_df)}")
        print(f"   - Total image samples: {len(self.samples)}")
        print(f"   - {split} split samples: {len(self.split_samples)}")
        
    def _create_samples(self) -> List[Dict]:
        """Create list of samples with image paths and weights."""
        samples = []
        
        for _, row in self.bin_df.iterrows():
            bin_id = row['dna_bin']
            weight = row['mean_weight']
            specimen_count = row['specimen_count']
            
            # Get BIOSCAN processids for this BIN
            bioscan_processids = eval(row['bioscan_processids']) if isinstance(row['bioscan_processids'], str) else []
            
            # Find image files for these processids
            for processid in bioscan_processids:
                image_path = self._find_image_path(processid)
                if image_path and os.path.exists(image_path):
                    sample = {
                        'image_path': image_path,
                        'weight': weight,
                        'bin_id': bin_id,
                        'processid': processid,
                        'specimen_count': specimen_count
                    }
                    
                    if self.use_scale_features:
                        # Add scale features if available
                        sample.update({
                            'area_fraction': getattr(row, 'area_fraction', 1.0),
                            'image_measurement_value': getattr(row, 'image_measurement_value', 1.0),
                            'scale_factor': getattr(row, 'scale_factor', 1.0)
                        })
                    
                    samples.append(sample)
        
        print(f"✅ Found {len(samples)} valid image-weight pairs")
        
        if self.max_samples:
            samples = samples[:self.max_samples]
            print(f"🔄 Limited to {len(samples)} samples for testing")
        
        return samples
    
    def _find_image_path(self, processid: str) -> Optional[str]:
        """Find image file path for a given processid."""
        # Try different possible image locations in BIOSCAN structure
        possible_paths = [
            f"{self.bioscan_images_root}/cropped_256/train/**/{processid}.jpg",
            f"{self.bioscan_images_root}/cropped_256/val/**/{processid}.jpg", 
            f"{self.bioscan_images_root}/cropped_256/test/**/{processid}.jpg",
            f"{self.bioscan_images_root}/cropped_256/**/{processid}.jpg"
        ]
        
        for pattern in possible_paths:
            matches = glob(pattern, recursive=True)
            if matches:
                return matches[0]
        
        return None
    
    def _split_data(self) -> List[Dict]:
        """Split data into train/val/test."""
        # For now, use simple random split
        # TODO: Implement proper train/val/test splits based on BIN IDs
        
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
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Get a sample from the dataset.
        
        Returns:
            image: Preprocessed image tensor
            weight: Target biomass weight
            scale_features: Scale features if use_scale_features=True
        """
        try:
            sample = self.split_samples[idx]
            
            # Load image
            image = Image.open(sample['image_path']).convert('RGB')
            
            # Apply transformations
            if self.transform:
                image = self.transform(image)
            
            # Get weight
            weight = torch.tensor(sample['weight'], dtype=torch.float32)
            
            # Get scale features if requested
            if self.use_scale_features:
                scale_features = torch.tensor([
                    sample.get('area_fraction', 1.0),
                    sample.get('image_measurement_value', 1.0),
                    sample.get('scale_factor', 1.0)
                ], dtype=torch.float32)
                return image, weight, scale_features
            else:
                return image, weight, None
                
        except Exception as e:
            print(f"❌ Error loading sample {idx}: {e}")
            # Return zero tensors as fallback
            image = torch.zeros(3, 224, 224)
            weight = torch.tensor(0.0, dtype=torch.float32)
            scale_features = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32) if self.use_scale_features else None
            return image, weight, scale_features


class BiomassEstimator(nn.Module):
    """Basic CNN for biomass estimation."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 7 * 7, 1000),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1000, 100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, 1)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class BiomassEstimator_scale(nn.Module):
    """CNN with scale features for biomass estimation."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 7 * 7 + 3, 1000),  # +3 for scale features
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1000, 100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, 1)
        )
        
    def forward(self, x, scale_features=None):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        if scale_features is not None:
            x = torch.cat([x, scale_features], dim=1)
        x = self.classifier(x)
        return x


class BiomassEstimator_resnet(nn.Module):
    """ResNet-based model for biomass estimation."""
    def __init__(self):
        super().__init__()
        # Load ResNet18 backbone
        try:
            self.resnet = models.resnet18(weights='DEFAULT')
            print("✅ ResNet18 loaded with DEFAULT weights")
        except:
            try:
                self.resnet = models.resnet18(pretrained=True)
                print("✅ ResNet18 loaded with pretrained=True (fallback)")
            except:
                self.resnet = models.resnet18(pretrained=False)
                print("⚠️ ResNet18 loaded without pretrained weights")
        
        # Replace final layer for regression
        self.resnet.fc = nn.Sequential(
            nn.Linear(512, 1000),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1000, 100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, 1)
        )
        
    def forward(self, x):
        return self.resnet(x)


class DirectBiomassTrainer:
    """Training class for direct biomass estimation models."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set random seeds
        self.set_random_seeds(config.get('random_seed', 42))
        
        # Training history
        self.train_history = []
        self.val_history = []
        
        print(f"✅ Direct trainer initialized on device: {self.device}")
        
    def set_random_seeds(self, seed: int):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def create_model(self, model_type: str) -> nn.Module:
        """Create the specified model architecture."""
        if model_type == 'basic':
            model = BiomassEstimator()
        elif model_type == 'scale':
            model = BiomassEstimator_scale()
        elif model_type == 'resnet':
            model = BiomassEstimator_resnet()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model = model.to(self.device)
        print(f"✅ Created {model_type} model with {self.count_parameters(model):,} parameters")
        return model
    
    def count_parameters(self, model: nn.Module) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Create training and validation dataloaders."""
        # Image transformations
        train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create datasets
        train_dataset = DirectBiomassDataset(
            bin_mapping_csv=self.config['bin_mapping_csv'],
            bioscan_images_root=self.config['bioscan_images_root'],
            split="train",
            transform=train_transforms,
            use_scale_features=self.config['model_type'] == 'scale',
            max_samples=self.config.get('max_samples', None)
        )
        
        val_dataset = DirectBiomassDataset(
            bin_mapping_csv=self.config['bin_mapping_csv'],
            bioscan_images_root=self.config['bioscan_images_root'],
            split="val",
            transform=val_transforms,
            use_scale_features=self.config['model_type'] == 'scale',
            max_samples=self.config.get('max_samples', None)
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config.get('num_workers', 2),
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config.get('num_workers', 2),
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        print(f"✅ Created dataloaders: {len(train_dataset)} train, {len(val_dataset)} val samples")
        return train_loader, val_loader
    
    def train_epoch(self, model: nn.Module, train_loader: DataLoader, 
                   optimizer: optim.Optimizer, criterion: nn.Module) -> Dict[str, float]:
        """Train for one epoch."""
        model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        for batch_idx, batch_data in enumerate(train_loader):
            try:
                if len(batch_data) == 3:
                    images, weights, scale_features = batch_data
                    images = images.to(self.device)
                    weights = weights.to(self.device)
                    
                    if scale_features is not None and self.config['model_type'] == 'scale':
                        scale_features = scale_features.to(self.device)
                        predictions = model(images, scale_features)
                    else:
                        predictions = model(images)
                else:
                    images, weights = batch_data
                    images = images.to(self.device)
                    weights = weights.to(self.device)
                    predictions = model(images)
                
                # Compute loss
                loss = criterion(predictions.squeeze(), weights)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    print(f"  Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}")
                    
            except Exception as e:
                print(f"❌ Error in batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / max(num_batches, 1)
        return {'loss': avg_loss, 'mae': avg_loss}
    
    def validate_epoch(self, model: nn.Module, val_loader: DataLoader, 
                      criterion: nn.Module) -> Dict[str, float]:
        """Validate for one epoch."""
        model.eval()
        total_loss = 0.0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(val_loader):
                try:
                    if len(batch_data) == 3:
                        images, weights, scale_features = batch_data
                        images = images.to(self.device)
                        weights = weights.to(self.device)
                        
                        if scale_features is not None and self.config['model_type'] == 'scale':
                            scale_features = scale_features.to(self.device)
                            predictions = model(images, scale_features)
                        else:
                            predictions = model(images)
                    else:
                        images, weights = batch_data
                        images = images.to(self.device)
                        weights = weights.to(self.device)
                        predictions = model(images)
                    
                    loss = criterion(predictions.squeeze(), weights)
                    total_loss += loss.item()
                    
                except Exception as e:
                    print(f"❌ Error in validation batch {batch_idx}: {e}")
                    continue
        
        avg_loss = total_loss / max(num_batches, 1)
        return {'loss': avg_loss, 'mae': avg_loss}
    
    def train(self):
        """Main training loop."""
        print("🚀 Starting direct biomass estimation training...")
        
        # Create model
        model = self.create_model(self.config['model_type'])
        
        # Create dataloaders
        train_loader, val_loader = self.create_dataloaders()
        
        if len(train_loader) == 0 or len(val_loader) == 0:
            print("❌ No valid samples found for training!")
            return
        
        # Setup optimizer and criterion
        optimizer = optim.Adam(model.parameters(), lr=self.config['learning_rate'])
        criterion = nn.L1Loss()  # Mean Absolute Error
        
        # Setup learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Training loop
        best_val_mae = float('inf')
        best_model_state = None
        
        for epoch in range(self.config['num_epochs']):
            print(f"\n🔄 Epoch {epoch + 1}/{self.config['num_epochs']}")
            
            # Train
            train_metrics = self.train_epoch(model, train_loader, optimizer, criterion)
            
            # Validate
            val_metrics = self.validate_epoch(model, val_loader, criterion)
            
            # Update learning rate
            scheduler.step(val_metrics['mae'])
            
            # Save best model
            if val_metrics['mae'] < best_val_mae:
                best_val_mae = val_metrics['mae']
                best_model_state = model.state_dict().copy()
                print(f"✅ New best model: MAE = {best_val_mae:.4f}")
            
            # Log metrics
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, MAE: {train_metrics['mae']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, MAE: {val_metrics['mae']:.4f}")
            
            # Store history
            self.train_history.append(train_metrics)
            self.val_history.append(val_metrics)
        
        # Save final model
        self.save_model(model, best_model_state)
        print(f"\n✅ Training completed! Best validation MAE: {best_val_mae:.4f}")
        
    def save_model(self, model: nn.Module, best_state_dict: Dict):
        """Save the trained model."""
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"direct_biomass_model_{self.config['model_type']}_{timestamp}.pth"
        
        model_path = output_dir / model_name
        torch.save({
            'model_state_dict': best_state_dict,
            'config': self.config,
            'train_history': self.train_history,
            'val_history': self.val_history
        }, model_path)
        
        print(f"✅ Model saved to: {model_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Direct Biomass Estimation Training')
    parser.add_argument('--bin_mapping_csv', type=str, required=True, help='Path to BIN mapping CSV')
    parser.add_argument('--bioscan_images_root', type=str, required=True, help='Path to BIOSCAN images root')
    parser.add_argument('--model_type', type=str, choices=['basic', 'scale', 'resnet'], 
                        default='resnet', help='Model architecture')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='./direct_model_outputs', help='Output directory')
    parser.add_argument('--max_samples', type=int, default=None, help='Max samples for testing')
    
    args = parser.parse_args()
    
    config = {
        'bin_mapping_csv': args.bin_mapping_csv,
        'bioscan_images_root': args.bioscan_images_root,
        'model_type': args.model_type,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'output_dir': args.output_dir,
        'max_samples': args.max_samples,
        'random_seed': 42,
        'num_workers': 2
    }
    
    print("🎯 Direct Biomass Estimation Training Pipeline")
    print("=" * 50)
    for key, value in config.items():
        print(f"{key}: {value}")
    print("=" * 50)
    
    trainer = DirectBiomassTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()