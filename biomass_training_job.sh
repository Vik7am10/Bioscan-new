#!/usr/bin/env python3
"""
Biomass ML Pipeline - Improved Training Script
Combines existing model architectures with modern data loading and training practices.
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
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchmetrics import MeanAbsoluteError, MeanAbsolutePercentageError
import h5py
from PIL import Image

# Import existing model architectures
sys.path.append('/mnt/c/Users/dogbo/Biomass5M/Biomass-5M-main')
try:
    from models import BiomassEstimator, BiomassEstimator_scale, BiomassEstimator_resnet
    print("✅ Successfully imported existing model architectures")
except ImportError as e:
    print(f"❌ Failed to import existing models: {e}")
    print("Please ensure the models.py file is in the correct location")
    sys.exit(1)


class ImprovedBioMassDataSet(Dataset):
    """
    Improved version of the biomass dataset with better error handling and performance.
    """
    
    def __init__(self, 
                 data_file: str,
                 biomass_csv: str,
                 transform: Optional[transforms.Compose] = None,
                 use_scale_features: bool = True,
                 train_mode: bool = True):
        """
        Initialize the biomass dataset.
        
        Args:
            data_file: Path to HDF5 file containing images
            biomass_csv: Path to CSV file with biomass data
            transform: Image transformations
            use_scale_features: Whether to include scale features
            train_mode: Whether this is for training (affects data splits)
        """
        self.data_file = data_file
        self.biomass_csv = biomass_csv
        self.transform = transform or transforms.Compose([transforms.ToTensor()])
        self.use_scale_features = use_scale_features
        self.train_mode = train_mode
        
        # Load biomass data
        self.biomass_df = self._load_biomass_data()
        
        # Validate data file exists
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        # Get dataset size
        with h5py.File(data_file, 'r') as f:
            self.length = len(f['images'])
            self.image_shape = f['images'].shape[1:]
            
        print(f"✅ Dataset initialized: {self.length} samples, image shape: {self.image_shape}")
        
    def _load_biomass_data(self) -> pd.DataFrame:
        """Load and validate biomass data."""
        try:
            # Try comma-separated first, then tab-separated
            try:
                df = pd.read_csv(self.biomass_csv, sep=',')
            except:
                df = pd.read_csv(self.biomass_csv, sep='\t')
            
            # Standardize column names
            column_mapping = {
                'index': 'processid',
                'weight_avg': 'weight',
                'processid': 'processid',
                'weight': 'weight'
            }
            
            for old_name, new_name in column_mapping.items():
                if old_name in df.columns and new_name not in df.columns:
                    df = df.rename(columns={old_name: new_name})
            
            # Validate required columns
            required_cols = ['processid', 'weight']
            if self.use_scale_features:
                required_cols.extend(['area_fraction', 'image_measurement_value', 'scale_factor'])
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Create processid to index mapping
            self.processid_to_idx = {pid: idx for idx, pid in enumerate(df['processid'])}
            
            print(f"✅ Loaded biomass data: {len(df)} records")
            return df
            
        except Exception as e:
            raise RuntimeError(f"Failed to load biomass data from {self.biomass_csv}: {e}")
    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Get a sample from the dataset.
        
        Returns:
            image: Preprocessed image tensor
            weight: Target biomass weight
            scale_features: Scale features if use_scale_features=True
        """
        try:
            # Load image from HDF5
            with h5py.File(self.data_file, 'r') as f:
                image_data = f['images'][idx]
                processid = f['processids'][idx]
                
            # Convert to PIL Image
            if image_data.dtype != np.uint8:
                image_data = (image_data * 255).astype(np.uint8)
            
            image = Image.fromarray(image_data)
            
            # Apply transformations
            if self.transform:
                image = self.transform(image)
            
            # Get biomass data
            if processid in self.processid_to_idx:
                biomass_idx = self.processid_to_idx[processid]
                biomass_row = self.biomass_df.iloc[biomass_idx]
                
                weight = torch.tensor(biomass_row['weight'], dtype=torch.float32)
                
                if self.use_scale_features:
                    scale_features = torch.tensor([
                        biomass_row['area_fraction'],
                        biomass_row['image_measurement_value'],
                        biomass_row['scale_factor']
                    ], dtype=torch.float32)
                    return image, weight, scale_features
                else:
                    return image, weight, None
            else:
                # Handle missing processid
                weight = torch.tensor(0.0, dtype=torch.float32)
                if self.use_scale_features:
                    scale_features = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
                    return image, weight, scale_features
                else:
                    return image, weight, None
                    
        except Exception as e:
            print(f"❌ Error loading sample {idx}: {e}")
            # Return zero tensors as fallback
            image = torch.zeros(3, 224, 224)  # Default image size
            weight = torch.tensor(0.0, dtype=torch.float32)
            scale_features = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32) if self.use_scale_features else None
            return image, weight, scale_features


class BiomassTrainer:
    """
    Main training class for biomass estimation models.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the trainer with configuration.
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set random seeds for reproducibility
        self.set_random_seeds(config.get('random_seed', 42))
        
        # Initialize metrics
        self.train_mae = MeanAbsoluteError()
        self.train_mape = MeanAbsolutePercentageError()
        self.val_mae = MeanAbsoluteError()
        self.val_mape = MeanAbsolutePercentageError()
        
        # Training history
        self.train_history = []
        self.val_history = []
        
        print(f"✅ Trainer initialized on device: {self.device}")
        
    def set_random_seeds(self, seed: int):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        print(f"✅ Random seeds set to {seed}")
    
    def create_model(self, model_type: str, num_classes: int = 1) -> nn.Module:
        """
        Create the specified model architecture.
        
        Args:
            model_type: Type of model ('basic', 'scale', 'resnet')
            num_classes: Number of output classes (1 for regression)
            
        Returns:
            PyTorch model
        """
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
        """Count the number of trainable parameters in a model."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Create training and validation dataloaders.
        
        Returns:
            Tuple of (train_loader, val_loader)
        """
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
        
        # Create full dataset
        full_dataset = ImprovedBioMassDataSet(
            data_file=self.config['data_file'],
            biomass_csv=self.config['biomass_csv'],
            transform=train_transforms,
            use_scale_features=self.config['model_type'] == 'scale'
        )
        
        # Split dataset
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        train_dataset, val_dataset = random_split(
            full_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.config.get('random_seed', 42))
        )
        
        # Update validation dataset transforms
        val_dataset.dataset.transform = val_transforms
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        print(f"✅ Created dataloaders: {len(train_dataset)} train, {len(val_dataset)} val samples")
        return train_loader, val_loader
    
    def train_epoch(self, model: nn.Module, train_loader: DataLoader, 
                   optimizer: optim.Optimizer, criterion: nn.Module) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary with training metrics
        """
        model.train()
        self.train_mae.reset()
        self.train_mape.reset()
        
        total_loss = 0.0
        num_batches = len(train_loader)
        
        for batch_idx, batch_data in enumerate(train_loader):
            if len(batch_data) == 3:
                images, weights, scale_features = batch_data
                images = images.to(self.device)
                weights = weights.to(self.device)
                
                if scale_features is not None:
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
            
            # Update metrics
            total_loss += loss.item()
            self.train_mae.update(predictions.squeeze(), weights)
            self.train_mape.update(predictions.squeeze(), weights)
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}")
        
        return {
            'loss': total_loss / num_batches,
            'mae': self.train_mae.compute().item(),
            'mape': self.train_mape.compute().item()
        }
    
    def validate_epoch(self, model: nn.Module, val_loader: DataLoader, 
                      criterion: nn.Module) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Returns:
            Dictionary with validation metrics
        """
        model.eval()
        self.val_mae.reset()
        self.val_mape.reset()
        
        total_loss = 0.0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for batch_data in val_loader:
                if len(batch_data) == 3:
                    images, weights, scale_features = batch_data
                    images = images.to(self.device)
                    weights = weights.to(self.device)
                    
                    if scale_features is not None:
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
                
                # Update metrics
                total_loss += loss.item()
                self.val_mae.update(predictions.squeeze(), weights)
                self.val_mape.update(predictions.squeeze(), weights)
        
        return {
            'loss': total_loss / num_batches,
            'mae': self.val_mae.compute().item(),
            'mape': self.val_mape.compute().item()
        }
    
    def train(self):
        """
        Main training loop.
        """
        print("🚀 Starting biomass estimation training...")
        
        # Create model
        model = self.create_model(self.config['model_type'])
        
        # Create dataloaders
        train_loader, val_loader = self.create_dataloaders()
        
        # Setup optimizer and criterion
        optimizer = optim.Adam(model.parameters(), lr=self.config['learning_rate'])
        criterion = nn.L1Loss()  # Mean Absolute Error for regression
        
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
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, MAE: {train_metrics['mae']:.4f}, MAPE: {train_metrics['mape']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, MAE: {val_metrics['mae']:.4f}, MAPE: {val_metrics['mape']:.4f}")
            
            # Store history
            self.train_history.append(train_metrics)
            self.val_history.append(val_metrics)
        
        # Save final model
        self.save_model(model, best_model_state)
        
        print(f"\n✅ Training completed! Best validation MAE: {best_val_mae:.4f}")
        
    def save_model(self, model: nn.Module, best_state_dict: Dict):
        """Save the trained model and training history."""
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"biomass_model_{self.config['model_type']}_{timestamp}.pth"
        
        # Save model
        model_path = output_dir / model_name
        torch.save({
            'model_state_dict': best_state_dict,
            'config': self.config,
            'train_history': self.train_history,
            'val_history': self.val_history
        }, model_path)
        
        # Save training history
        history_path = output_dir / f"training_history_{timestamp}.json"
        with open(history_path, 'w') as f:
            json.dump({
                'train_history': self.train_history,
                'val_history': self.val_history,
                'config': self.config
            }, f, indent=2)
        
        print(f"✅ Model saved to: {model_path}")
        print(f"✅ Training history saved to: {history_path}")


def main():
    """Main function to run the training pipeline."""
    parser = argparse.ArgumentParser(description='Biomass Estimation Training Pipeline')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--data_file', type=str, required=True, help='Path to HDF5 data file')
    parser.add_argument('--biomass_csv', type=str, required=True, help='Path to biomass CSV file')
    parser.add_argument('--model_type', type=str, choices=['basic', 'scale', 'resnet'], 
                        default='resnet', help='Model architecture to use')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='./model_outputs', help='Output directory')
    
    args = parser.parse_args()
    
    # Create configuration
    config = {
        'data_file': args.data_file,
        'biomass_csv': args.biomass_csv,
        'model_type': args.model_type,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'output_dir': args.output_dir,
        'random_seed': 42,
        'num_workers': 4
    }
    
    # Load config file if provided
    if args.config:
        with open(args.config, 'r') as f:
            file_config = json.load(f)
        config.update(file_config)
    
    print("🎯 Biomass Estimation Training Pipeline")
    print("=" * 50)
    print(f"Model type: {config['model_type']}")
    print(f"Data file: {config['data_file']}")
    print(f"Biomass CSV: {config['biomass_csv']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Epochs: {config['num_epochs']}")
    print(f"Learning rate: {config['learning_rate']}")
    print("=" * 50)
    
    # Create trainer and start training
    trainer = BiomassTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()