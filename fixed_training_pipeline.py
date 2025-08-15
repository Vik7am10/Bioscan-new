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
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from PIL import Image
import torchvision.models as models
from sklearn.model_selection import StratifiedKFold

class MAPELoss(nn.Module):
    """Mean Absolute Percentage Error Loss for biomass prediction."""
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, predictions, targets):
        # Denormalize predictions and targets to real values
        # Assuming targets are already in real mg values or appropriately scaled
        
        # Add epsilon to avoid division by zero
        targets_safe = torch.abs(targets) + self.epsilon
        
        # Calculate MAPE: mean(|actual - predicted| / |actual|) * 100
        mape = torch.mean(torch.abs(predictions - targets) / targets_safe) * 100
        
        return mape

class SymmetricMAPELoss(nn.Module):
    """Symmetric Mean Absolute Percentage Error Loss - more robust."""
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, predictions, targets):
        # SMAPE: 2 * mean(|actual - predicted| / (|actual| + |predicted|)) * 100
        numerator = torch.abs(predictions - targets)
        denominator = torch.abs(targets) + torch.abs(predictions) + self.epsilon
        
        smape = torch.mean(2 * numerator / denominator) * 100
        
        return smape

class HybridLoss(nn.Module):
    """Combination of MAE and SMAPE for better training."""
    def __init__(self, mae_weight=0.3, smape_weight=0.7, epsilon=1e-8):
        super().__init__()
        self.mae_weight = mae_weight
        self.smape_weight = smape_weight
        self.mae_loss = nn.L1Loss()
        self.smape_loss = SymmetricMAPELoss(epsilon)
    
    def forward(self, predictions, targets):
        mae = self.mae_loss(predictions, targets)
        smape = self.smape_loss(predictions, targets)
        
        # Scale SMAPE to similar magnitude as MAE for balanced training
        return self.mae_weight * mae + self.smape_weight * (smape / 100.0)

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
                 target_scaler: Optional[Dict] = None,
                 use_cleaned: bool = True,
                 config: Optional[Dict] = None):
        """
        Initialize dataset with target normalization.
        
        Args:
            bin_mapping_csv: Path to BIN mapping results CSV
            image_index_json: Path to pre-built image index JSON
            split: Dataset split ('train', 'val', 'test')
            transform: Image transformations
            max_samples: Maximum number of samples to use
            target_scaler: Dict with 'mean' and 'std' for target normalization
            config: Configuration dict with filtering options
        """
        self.bin_mapping_csv = bin_mapping_csv
        self.image_index_json = image_index_json
        self.split = split
        self.transform = transform
        self.max_samples = max_samples
        self.target_scaler = target_scaler
        self.use_cleaned = use_cleaned
        self.config = config or {}
        
        # Load image index
        print(f" Loading image index from {image_index_json}...")
        with open(image_index_json, 'r') as f:
            self.image_index = json.load(f)
        print(f" Loaded index with {len(self.image_index)} images")
        
        # Load BIN mapping data - use cleaned version if available and requested
        if use_cleaned and os.path.exists('bin_results/bin_id_biomass_mapping_cleaned.csv'):
            bin_mapping_path = 'bin_results/bin_id_biomass_mapping_cleaned.csv'
            print(f" Loading CLEANED BIN mapping data from {bin_mapping_path}...")
        else:
            bin_mapping_path = bin_mapping_csv
            print(f" Loading original BIN mapping data from {bin_mapping_path}...")
            
        self.bin_df = pd.read_csv(bin_mapping_path)
        print(f" Loaded {len(self.bin_df)} BIN mappings")
        
        # Apply dataset filters if requested
        if self.config.get('filter_outliers') or self.config.get('max_biomass') is not None:
            print(f"\n🔧 Applying dataset filters...")
            self.bin_df = apply_dataset_filters(self.bin_df, self.config)
            print("")
        
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
        """Split data into train/val/test (70/15/15)."""
        # Use fixed seed for reproducible splits
        random.seed(42)
        random.shuffle(self.samples)
        
        n_samples = len(self.samples)
        train_end = int(0.7 * n_samples)  # 70% for training
        val_end = int(0.85 * n_samples)   # 15% for validation
        # Remaining 15% for test
        
        if self.split == "train":
            return self.samples[:train_end]
        elif self.split == "val":
            return self.samples[train_end:val_end]
        elif self.split == "test":
            return self.samples[val_end:]
        else:
            raise ValueError(f"Unknown split: {self.split}")
    
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


def get_loss_function(loss_name: str):
    """Get loss function by name."""
    loss_functions = {
        'mae': nn.L1Loss(),
        'mse': nn.MSELoss(), 
        'mape': MAPELoss(),
        'smape': SymmetricMAPELoss(),
        'hybrid': HybridLoss()
    }
    
    if loss_name.lower() not in loss_functions:
        raise ValueError(f"Unknown loss function: {loss_name}. Available: {list(loss_functions.keys())}")
    
    return loss_functions[loss_name.lower()]


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


def filter_outliers_scaling_law(df, target_col='mean_weight', area_col='mean_pixel_area'):
    """Filter outliers using biological scaling law analysis."""
    from sklearn.linear_model import HuberRegressor
    
    print(" Applying scaling law outlier filtering...")
    
    # Calculate scaling features
    sqrt_area = np.sqrt(df[area_col].values)
    cube_root_mass = np.cbrt(df[target_col].values)
    
    # Fit robust regression model
    scaling_model = HuberRegressor(epsilon=1.35)
    scaling_model.fit(sqrt_area.reshape(-1, 1), cube_root_mass)
    
    # Predict expected cube_root_mass from sqrt_area
    expected_cube_root_mass = scaling_model.predict(sqrt_area.reshape(-1, 1))
    scaling_coefficient = scaling_model.coef_[0]
    scaling_intercept = scaling_model.intercept_
    
    print(f"   Scaling relationship: ∛(mass) = {scaling_coefficient:.6f} * √(area) + {scaling_intercept:.6f}")
    
    # Calculate deviations from scaling law
    scaling_residuals = cube_root_mass - expected_cube_root_mass
    
    # Identify outliers using both absolute and relative thresholds
    # Method 1: Statistical outliers (3-sigma rule)
    abs_threshold = 3 * np.std(scaling_residuals)
    abs_outlier_mask = np.abs(scaling_residuals) > abs_threshold
    
    # Method 2: Relative outliers (mass significantly different from scaling prediction)
    predicted_mass = expected_cube_root_mass**3
    relative_error = np.abs(df[target_col].values - predicted_mass) / (predicted_mass + 1e-6)
    rel_threshold = np.percentile(relative_error, 95)  # Top 5% relative errors
    rel_outlier_mask = relative_error > rel_threshold
    
    # Combined outlier detection (either method flags it)
    outlier_mask = abs_outlier_mask | rel_outlier_mask
    
    # Filter out outliers
    df_filtered = df[~outlier_mask].copy()
    
    print(f"   Removed {outlier_mask.sum()} outliers ({outlier_mask.sum()/len(df)*100:.1f}%)")
    print(f"   Remaining samples: {len(df_filtered)}")
    
    # Show scaling law validation
    correlation_sqrt_cbrt = np.corrcoef(sqrt_area[~outlier_mask], cube_root_mass[~outlier_mask])[0,1]
    print(f"   Scaling correlation (after filtering): {correlation_sqrt_cbrt:.4f}")
    
    return df_filtered


def apply_dataset_filters(df, config):
    """Apply all dataset filters based on configuration."""
    original_size = len(df)
    print(f" Original dataset size: {original_size}")
    
    # Filter by maximum biomass if specified
    if config.get('max_biomass') is not None:
        max_biomass = config['max_biomass']
        df = df[df['mean_weight'] <= max_biomass].copy()
        removed = original_size - len(df)
        print(f" After max biomass filter (≤{max_biomass}mg): {len(df)} samples (-{removed})")
    
    # Filter outliers using scaling law if requested
    if config.get('filter_outliers'):
        # Check if we have the required columns for scaling law analysis
        if 'mean_pixel_area' in df.columns or 'pixel_area' in df.columns:
            area_col = 'mean_pixel_area' if 'mean_pixel_area' in df.columns else 'pixel_area'
            weight_col = 'mean_weight' if 'mean_weight' in df.columns else 'weight'
            df = filter_outliers_scaling_law(df, target_col=weight_col, area_col=area_col)
        else:
            print("   Warning: Cannot apply scaling law filtering - missing pixel area column")
    
    total_removed = original_size - len(df)
    if total_removed > 0:
        print(f" Total samples removed: {total_removed} ({total_removed/original_size*100:.1f}%)")
        print(f" Final dataset size: {len(df)}")
    
    return df


def create_stratified_folds(targets, n_splits=5, random_state=42):
    """Create stratified K-fold splits based on biomass quantiles."""
    # Create biomass bins for stratification (uniform distribution across folds)
    quantiles = np.quantile(targets, np.linspace(0, 1, 11))  # 10 bins
    bins = np.digitize(targets, quantiles[1:-1])
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_indices = list(skf.split(range(len(targets)), bins))
    
    print(f" Created {n_splits} stratified folds:")
    for i, (train_idx, val_idx) in enumerate(fold_indices):
        train_weights = targets[train_idx]
        val_weights = targets[val_idx]
        print(f"   Fold {i+1}: Train={len(train_idx)} (mean={train_weights.mean():.3f}), "
              f"Val={len(val_idx)} (mean={val_weights.mean():.3f})")
    
    return fold_indices


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
        target_scaler=target_scaler,
        use_cleaned=config.get('use_cleaned', False),
        config=config
    )
    
    val_dataset = FixedBiomassDataset(
        bin_mapping_csv=config['bin_mapping_csv'],
        image_index_json=config['image_index_json'],
        split='val',
        transform=val_transforms,
        max_samples=config.get('max_samples'),
        target_scaler=target_scaler,
        use_cleaned=config.get('use_cleaned', False),
        config=config
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
    
    # Loss function selection
    loss_type = config.get('loss_function', 'smape')  # Default to SMAPE
    criterion = get_loss_function(loss_type)
    print(f" Using {loss_type.upper()} Loss")
    
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
            
            # For MAPE losses, denormalize to real mg values
            if loss_type in ['mape', 'smape', 'hybrid']:
                # Denormalize predictions and targets
                real_outputs = outputs * target_scaler['std'] + target_scaler['mean']
                real_targets = targets * target_scaler['std'] + target_scaler['mean']
                loss = criterion(real_outputs, real_targets)
            else:
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
                
                # For MAPE losses, denormalize to real mg values
                if loss_type in ['mape', 'smape', 'hybrid']:
                    real_outputs = outputs * target_scaler['std'] + target_scaler['mean']
                    real_targets = targets * target_scaler['std'] + target_scaler['mean']
                    loss = criterion(real_outputs, real_targets)
                else:
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


def train_kfold(config: Dict):
    """Train model using K-fold cross-validation."""
    print(f"\n🔄 Starting {config['k_folds']}-Fold Cross-Validation Training")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    torch.manual_seed(config.get('random_seed', 42))
    np.random.seed(config.get('random_seed', 42))
    random.seed(config.get('random_seed', 42))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" Using device: {device}")
    
    # Calculate target scaler
    target_scaler = calculate_target_scaler(config['bin_mapping_csv'])
    
    # Create full dataset
    transforms_config = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    full_dataset = FixedBiomassDataset(
        bin_mapping_csv=config['bin_mapping_csv'],
        image_index_json=config['image_index_json'],
        split='train',  # Use all available data
        transform=transforms_config,
        max_samples=config.get('max_samples'),
        target_scaler=target_scaler,
        use_cleaned=config.get('use_cleaned', False),
        config=config
    )
    
    # Get targets for stratification
    targets = np.array([full_dataset[i][1].item() * target_scaler['std'] + target_scaler['mean'] 
                       for i in range(len(full_dataset))])
    
    # Create stratified folds
    fold_indices = create_stratified_folds(targets, n_splits=config['k_folds'])
    
    # Store results for each fold
    fold_results = []
    best_models = []
    
    for fold, (train_idx, val_idx) in enumerate(fold_indices):
        print(f"\n📊 Training Fold {fold + 1}/{config['k_folds']}")
        print("-" * 40)
        
        # Create data loaders for this fold
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        
        train_loader = DataLoader(
            full_dataset,
            batch_size=config['batch_size'],
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            full_dataset,
            batch_size=config['batch_size'],
            sampler=val_sampler,
            num_workers=4,
            pin_memory=True
        )
        
        # Create model for this fold
        model = create_model(dropout_rate=0.3)
        model = model.to(device)
        
        # Loss function and optimizer
        loss_type = config.get('loss_function', 'smape')
        criterion = get_loss_function(loss_type)
        optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        # Training loop for this fold
        best_val_loss = float('inf')
        patience_counter = 0
        patience_limit = 15
        
        train_losses = []
        val_losses = []
        
        for epoch in range(config['num_epochs']):
            # Training phase
            model.train()
            train_loss = 0.0
            train_batches = 0
            
            for batch_idx, (images, targets_batch) in enumerate(train_loader):
                images = images.to(device)
                targets_batch = targets_batch.to(device)
                
                optimizer.zero_grad()
                outputs = model(images).squeeze()
                
                # Apply loss function
                if loss_type in ['mape', 'smape', 'hybrid']:
                    # Denormalize for MAPE-based losses
                    real_outputs = outputs * target_scaler['std'] + target_scaler['mean']
                    real_targets = targets_batch * target_scaler['std'] + target_scaler['mean']
                    loss = criterion(real_outputs, real_targets)
                else:
                    loss = criterion(outputs, targets_batch)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1
            
            avg_train_loss = train_loss / train_batches
            train_losses.append(avg_train_loss)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for images, targets_batch in val_loader:
                    images = images.to(device)
                    targets_batch = targets_batch.to(device)
                    
                    outputs = model(images).squeeze()
                    
                    if loss_type in ['mape', 'smape', 'hybrid']:
                        real_outputs = outputs * target_scaler['std'] + target_scaler['mean']
                        real_targets = targets_batch * target_scaler['std'] + target_scaler['mean']
                        loss = criterion(real_outputs, real_targets)
                    else:
                        loss = criterion(outputs, targets_batch)
                    
                    val_loss += loss.item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches
            val_losses.append(avg_val_loss)
            
            if epoch % 5 == 0:
                print(f"  Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model for this fold
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= patience_limit:
                print(f"  Early stopping after {patience_limit} epochs without improvement")
                break
        
        # Store results for this fold
        fold_results.append({
            'fold': fold + 1,
            'best_val_loss': best_val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'final_epoch': epoch + 1
        })
        
        # Store best model
        best_models.append(best_model_state)
        
        print(f"  Fold {fold + 1} completed - Best Val Loss: {best_val_loss:.4f}")
    
    # Calculate cross-validation statistics
    val_losses = [result['best_val_loss'] for result in fold_results]
    mean_val_loss = np.mean(val_losses)
    std_val_loss = np.std(val_losses)
    
    print(f"\n📈 K-Fold Cross-Validation Results:")
    print("=" * 50)
    print(f" Mean Validation Loss: {mean_val_loss:.4f} ± {std_val_loss:.4f}")
    print(f" Individual Fold Results:")
    for result in fold_results:
        print(f"   Fold {result['fold']}: {result['best_val_loss']:.4f} (epochs: {result['final_epoch']})")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"{config['output_dir']}/kfold_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save individual fold models
    for i, model_state in enumerate(best_models):
        model_path = f"{results_dir}/fold_{i+1}_model.pth"
        torch.save({
            'model_state_dict': model_state,
            'config': config,
            'target_scaler': target_scaler,
            'fold_results': fold_results[i]
        }, model_path)
    
    # Save overall results
    results_summary = {
        'config': config,
        'target_scaler': target_scaler,
        'fold_results': fold_results,
        'mean_val_loss': mean_val_loss,
        'std_val_loss': std_val_loss,
        'loss_function': loss_type
    }
    
    results_path = f"{results_dir}/kfold_results.json"
    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    print(f"\n💾 K-Fold results saved to: {results_dir}")
    
    return results_dir, mean_val_loss, std_val_loss


def main():
    parser = argparse.ArgumentParser(description='Enhanced Biomass Training Pipeline')
    parser.add_argument('--bin_mapping_csv', default='bin_results/bin_id_biomass_mapping.csv',
                       help='Path to BIN mapping CSV file')
    parser.add_argument('--image_index_json', default='image_index.json',
                       help='Path to image index JSON file')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                       help='Learning rate')
    parser.add_argument('--output_dir', default='enhanced_model_outputs',
                       help='Output directory for model and results')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to use (None for all)')
    
    # New loss function argument
    parser.add_argument('--loss_function', choices=['mae', 'mse', 'mape', 'smape', 'hybrid'], 
                       default='smape', help='Loss function to use (default: smape)')
    
    # K-fold cross-validation arguments
    parser.add_argument('--use_kfold', action='store_true',
                       help='Use K-fold cross-validation instead of train/val split')
    parser.add_argument('--k_folds', type=int, default=5,
                       help='Number of folds for K-fold CV (default: 5)')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # Dataset filtering options
    parser.add_argument('--use_cleaned', action='store_true',
                       help='Use cleaned dataset (filtered outliers)')
    parser.add_argument('--filter_outliers', action='store_true',
                       help='Filter outliers on-the-fly using scaling law analysis')
    parser.add_argument('--max_biomass', type=float, default=None,
                       help='Filter out samples above this biomass threshold (in mg)')
    
    args = parser.parse_args()
    
    # Convert args to config dict
    config = vars(args)
    
    # Print configuration
    print(" Enhanced Biomass Training Configuration:")
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
    
    print(" Required files found.")
    
    # Choose training method
    if args.use_kfold:
        print(f" Starting {args.k_folds}-fold cross-validation training...")
        results_dir, mean_loss, std_loss = train_kfold(config)
        print(f"\n✅ K-Fold training completed!")
        print(f" Results saved to: {results_dir}")
        print(f" Mean validation loss: {mean_loss:.4f} ± {std_loss:.4f}")
    else:
        print(" Starting single train/val split training...")
        model_path = train_model(config)
        print(f"\n✅ Single training completed!")
        print(f" Model saved to: {model_path}")

if __name__ == "__main__":
    main()