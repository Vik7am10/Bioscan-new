#!/usr/bin/env python3
"""
K-means 5-fold cross-validation for robust biomass prediction evaluation.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error

from fixed_training_pipeline import (
    FixedBiomassDataset, 
    ImprovedBiomassEstimator, 
    calculate_target_scaler,
    train_model
)

class KMeansCrossValidator:
    """K-means based cross-validation for biomass prediction."""
    
    def __init__(self, 
                 bin_mapping_csv: str,
                 image_index_json: str,
                 n_folds: int = 5,
                 use_cleaned: bool = True):
        """
        Initialize K-means cross-validator.
        
        Args:
            bin_mapping_csv: Path to BIN mapping CSV
            image_index_json: Path to image index JSON
            n_folds: Number of cross-validation folds
            use_cleaned: Whether to use cleaned dataset
        """
        self.bin_mapping_csv = bin_mapping_csv
        self.image_index_json = image_index_json
        self.n_folds = n_folds
        self.use_cleaned = use_cleaned
        
        # Load data
        print(f"🔄 Loading data for {n_folds}-fold cross-validation...")
        self._load_data()
        
        # Create k-means clusters
        print(f"🎯 Creating {n_folds} k-means clusters...")
        self._create_clusters()
        
    def _load_data(self):
        """Load BIN mapping and image data."""
        # Load BIN mapping
        if self.use_cleaned and os.path.exists('bin_results/bin_id_biomass_mapping_cleaned.csv'):
            self.bin_df = pd.read_csv('bin_results/bin_id_biomass_mapping_cleaned.csv')
            print(f"   📊 Loaded cleaned BIN mapping: {len(self.bin_df)} BINs")
        else:
            self.bin_df = pd.read_csv(self.bin_mapping_csv)
            print(f"   📊 Loaded original BIN mapping: {len(self.bin_df)} BINs")
            
        # Load image index
        with open(self.image_index_json, 'r') as f:
            self.image_index = json.load(f)
        print(f"   📸 Loaded image index: {len(self.image_index)} images")
        
        # Create samples list for clustering
        self.samples = []
        for _, row in self.bin_df.iterrows():
            bin_id = row['dna_bin']
            weight = row['mean_weight']
            
            # Get bioscan processids
            try:
                if isinstance(row['bioscan_processids'], str):
                    bioscan_processids = eval(row['bioscan_processids'])
                else:
                    bioscan_processids = row['bioscan_processids']
            except:
                bioscan_processids = []
            
            # Add valid samples
            for processid in bioscan_processids:
                if processid in self.image_index:
                    image_path = self.image_index[processid]
                    if os.path.exists(image_path):
                        self.samples.append({
                            'processid': processid,
                            'bin_id': bin_id,
                            'weight': weight,
                            'image_path': image_path
                        })
        
        print(f"   ✅ Created {len(self.samples)} valid samples")
        
    def _create_clusters(self):
        """Create k-means clusters based on biomass values."""
        # Extract weights for clustering
        weights = np.array([sample['weight'] for sample in self.samples])
        
        # Reshape for sklearn
        weights_reshaped = weights.reshape(-1, 1)
        
        # Perform k-means clustering
        kmeans = KMeans(n_clusters=self.n_folds, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(weights_reshaped)
        
        # Add cluster labels to samples
        for i, sample in enumerate(self.samples):
            sample['cluster'] = cluster_labels[i]
        
        # Print cluster statistics
        print(f"   📈 K-means clustering results:")
        for cluster_id in range(self.n_folds):
            cluster_samples = [s for s in self.samples if s['cluster'] == cluster_id]
            cluster_weights = [s['weight'] for s in cluster_samples]
            print(f"     Cluster {cluster_id}: {len(cluster_samples)} samples, "
                  f"weight range: {min(cluster_weights):.2f}-{max(cluster_weights):.2f} mg")
        
        self.cluster_labels = cluster_labels
        self.kmeans_model = kmeans
        
    def create_fold_splits(self) -> List[Tuple[List[int], List[int]]]:
        """
        Create train/val splits for each fold.
        
        Returns:
            List of (train_indices, val_indices) tuples
        """
        folds = []
        
        for fold in range(self.n_folds):
            # Validation set: samples from current cluster
            val_indices = [i for i, sample in enumerate(self.samples) 
                          if sample['cluster'] == fold]
            
            # Training set: samples from all other clusters
            train_indices = [i for i, sample in enumerate(self.samples) 
                           if sample['cluster'] != fold]
            
            folds.append((train_indices, val_indices))
            
            print(f"   Fold {fold}: {len(train_indices)} train, {len(val_indices)} val")
        
        return folds
        
    def evaluate_fold(self, 
                     fold_id: int,
                     train_indices: List[int],
                     val_indices: List[int],
                     config: Dict) -> Dict:
        """
        Train and evaluate a single fold.
        
        Args:
            fold_id: Current fold number
            train_indices: Training sample indices
            val_indices: Validation sample indices
            config: Training configuration
            
        Returns:
            Dictionary with fold results
        """
        print(f"\n🎯 Evaluating Fold {fold_id}")
        print("=" * 40)
        
        # Create temporary samples files for this fold
        train_samples = [self.samples[i] for i in train_indices]
        val_samples = [self.samples[i] for i in val_indices]
        
        # Calculate target scaler on training data only
        train_weights = [s['weight'] for s in train_samples]
        target_scaler = {
            'mean': np.mean(train_weights),
            'std': np.std(train_weights)
        }
        
        print(f"   Training samples: {len(train_samples)}")
        print(f"   Validation samples: {len(val_samples)}")
        print(f"   Target scaler - Mean: {target_scaler['mean']:.2f}, Std: {target_scaler['std']:.2f}")
        
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
        
        # Create custom datasets for this fold
        train_dataset = FoldDataset(train_samples, train_transforms, target_scaler)
        val_dataset = FoldDataset(val_samples, val_transforms, target_scaler)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config['batch_size'], 
            shuffle=True, 
            num_workers=2
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config['batch_size'], 
            shuffle=False, 
            num_workers=2
        )
        
        # Train model for this fold
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = ImprovedBiomassEstimator(dropout_rate=0.3).to(device)
        
        # Training loop (simplified)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
        criterion = nn.HuberLoss(delta=1.0)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(config['num_epochs']):
            # Training
            model.train()
            train_loss = 0.0
            train_batches = 0
            
            for images, targets in train_loader:
                images, targets = images.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(images).squeeze()
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1
            
            avg_train_loss = train_loss / train_batches
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for images, targets in val_loader:
                    images, targets = images.to(device), targets.to(device)
                    outputs = model(images).squeeze()
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches
            
            if epoch % 5 == 0:
                print(f"     Epoch {epoch}: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= 5:  # Reduced patience for CV
                print(f"     Early stopping at epoch {epoch}")
                break
        
        # Final evaluation with MAE in mg
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images).squeeze()
                
                # Denormalize to real mg values
                real_outputs = outputs.cpu().numpy() * target_scaler['std'] + target_scaler['mean']
                real_targets = targets.cpu().numpy() * target_scaler['std'] + target_scaler['mean']
                
                all_predictions.extend(real_outputs)
                all_targets.extend(real_targets)
        
        # Calculate MAE in mg
        mae_mg = mean_absolute_error(all_targets, all_predictions)
        
        print(f"   ✅ Fold {fold_id} MAE: {mae_mg:.3f} mg")
        
        return {
            'fold_id': fold_id,
            'mae_mg': mae_mg,
            'best_val_loss': best_val_loss,
            'n_train': len(train_samples),
            'n_val': len(val_samples),
            'val_weight_range': (min([s['weight'] for s in val_samples]), 
                               max([s['weight'] for s in val_samples]))
        }
    
    def run_cross_validation(self, config: Dict) -> Dict:
        """
        Run complete k-means cross-validation.
        
        Args:
            config: Training configuration
            
        Returns:
            Dictionary with cross-validation results
        """
        print(f"\n🚀 Starting {self.n_folds}-fold K-means Cross-Validation")
        print("=" * 60)
        
        # Create fold splits
        folds = self.create_fold_splits()
        
        # Run each fold
        fold_results = []
        for fold_id, (train_indices, val_indices) in enumerate(folds):
            result = self.evaluate_fold(fold_id, train_indices, val_indices, config)
            fold_results.append(result)
        
        # Calculate overall statistics
        mae_scores = [r['mae_mg'] for r in fold_results]
        
        cv_results = {
            'fold_results': fold_results,
            'mean_mae': np.mean(mae_scores),
            'std_mae': np.std(mae_scores),
            'min_mae': np.min(mae_scores),
            'max_mae': np.max(mae_scores),
            'baseline_mae': 0.52,  # Established baseline
            'n_folds': self.n_folds,
            'total_samples': len(self.samples)
        }
        
        # Print final results
        print(f"\n📊 K-means {self.n_folds}-Fold Cross-Validation Results")
        print("=" * 60)
        print(f"   Mean MAE: {cv_results['mean_mae']:.3f} ± {cv_results['std_mae']:.3f} mg")
        print(f"   Range: {cv_results['min_mae']:.3f} - {cv_results['max_mae']:.3f} mg")
        print(f"   Baseline: {cv_results['baseline_mae']:.3f} mg")
        
        if cv_results['mean_mae'] < cv_results['baseline_mae']:
            improvement = ((cv_results['baseline_mae'] - cv_results['mean_mae']) / cv_results['baseline_mae']) * 100
            print(f"   ✅ BEATS BASELINE by {improvement:.1f}%")
        else:
            print(f"   ❌ Below baseline")
        
        print(f"\n   Individual fold results:")
        for result in fold_results:
            print(f"     Fold {result['fold_id']}: {result['mae_mg']:.3f} mg "
                  f"(weight range: {result['val_weight_range'][0]:.2f}-{result['val_weight_range'][1]:.2f} mg)")
        
        return cv_results

class FoldDataset(torch.utils.data.Dataset):
    """Custom dataset for cross-validation folds."""
    
    def __init__(self, samples: List[Dict], transform, target_scaler: Dict):
        self.samples = samples
        self.transform = transform
        self.target_scaler = target_scaler
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        try:
            sample = self.samples[idx]
            
            # Load image
            from PIL import Image
            image = Image.open(sample['image_path']).convert('RGB')
            if self.transform:
                image = self.transform(image)
            
            # Normalize weight
            weight = sample['weight']
            if self.target_scaler:
                weight = (weight - self.target_scaler['mean']) / self.target_scaler['std']
            
            return image, torch.tensor(weight, dtype=torch.float32)
            
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            # Return zero tensors as fallback
            image = torch.zeros(3, 224, 224)
            weight = torch.tensor(0.0, dtype=torch.float32)
            return image, weight

def main():
    """Run k-means cross-validation evaluation."""
    print("🎯 K-means 5-Fold Cross-Validation for Biomass Prediction")
    print("=" * 65)
    
    # Configuration
    config = {
        'batch_size': 32,
        'learning_rate': 0.0001,
        'num_epochs': 20,  # Reduced for CV speed
        'use_cleaned': True
    }
    
    # Initialize cross-validator
    cv = KMeansCrossValidator(
        bin_mapping_csv='bin_results/bin_id_biomass_mapping_cleaned.csv',
        image_index_json='image_index.json',
        n_folds=5,
        use_cleaned=True
    )
    
    # Run cross-validation
    results = cv.run_cross_validation(config)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"kmeans_cv_results_{timestamp}.json"
    
    # Convert numpy types for JSON serialization
    json_results = {
        'mean_mae': float(results['mean_mae']),
        'std_mae': float(results['std_mae']),
        'min_mae': float(results['min_mae']),
        'max_mae': float(results['max_mae']),
        'baseline_mae': float(results['baseline_mae']),
        'n_folds': results['n_folds'],
        'total_samples': results['total_samples'],
        'fold_results': [
            {
                'fold_id': r['fold_id'],
                'mae_mg': float(r['mae_mg']),
                'best_val_loss': float(r['best_val_loss']),
                'n_train': r['n_train'],
                'n_val': r['n_val'],
                'val_weight_range': [float(r['val_weight_range'][0]), float(r['val_weight_range'][1])]
            }
            for r in results['fold_results']
        ],
        'config': config,
        'timestamp': timestamp
    }
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    return results

if __name__ == "__main__":
    main()