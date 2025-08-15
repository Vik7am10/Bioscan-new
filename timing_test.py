#!/usr/bin/env python3
"""
Timing test: Run 1 epoch on cleaned dataset to estimate training time.
"""

import os
import time
import torch
from fixed_training_pipeline import train_model

def main():
    """Run 1 epoch timing test."""
    
    print("🕐 TIMING TEST: 1 Epoch on Cleaned Dataset")
    print("=" * 50)
    
    # Check if cleaned dataset exists
    cleaned_path = 'bin_results/bin_id_biomass_mapping_cleaned.csv'
    if not os.path.exists(cleaned_path):
        print(f"❌ Cleaned dataset not found: {cleaned_path}")
        return
    
    # Print dataset info
    import pandas as pd
    df = pd.read_csv(cleaned_path)
    total_samples = df['specimen_count'].sum()
    print(f"📊 Dataset: {len(df)} BIN groups, {total_samples:,} total samples")
    
    # Training configuration for timing test
    config = {
        'bin_mapping_csv': 'bin_results/bin_id_biomass_mapping_cleaned.csv',  # Use cleaned dataset
        'image_index_json': 'image_index.json',
        'max_samples': None,  # Use all samples
        'batch_size': 32,
        'learning_rate': 0.0001,
        'num_epochs': 1,  # Only 1 epoch for timing
        'patience': 999,  # No early stopping for timing test
        'val_split': 0.2,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_model': False,  # Don't save for timing test
        'seed': 42,
        'use_cleaned': True,  # Use cleaned dataset
        'output_dir': './models'  # Required by train_model
    }
    
    print(f"🔧 Configuration:")
    print(f"   Device: {config['device']}")
    print(f"   Batch size: {config['batch_size']}")
    print(f"   Learning rate: {config['learning_rate']}")
    print(f"   Epochs: {config['num_epochs']} (timing test)")
    print(f"   Use cleaned dataset: True")
    
    # Record start time
    start_time = time.time()
    print(f"\n⏰ Starting 1-epoch timing test at {time.strftime('%H:%M:%S')}")
    
    try:
        # Run training with config dict
        results = train_model(config)
        
        # Calculate timing
        end_time = time.time()
        epoch_time = end_time - start_time
        
        print(f"\n✅ TIMING TEST COMPLETED")
        print("=" * 50)
        print(f"⏱️  1 Epoch Time: {epoch_time:.1f} seconds ({epoch_time/60:.1f} minutes)")
        
        # Estimate full training time
        typical_epochs = 50  # Estimate for convergence
        early_stop_epochs = 30  # Conservative estimate with early stopping
        
        full_time_max = epoch_time * typical_epochs
        full_time_est = epoch_time * early_stop_epochs
        
        print(f"\n📊 TIME ESTIMATES:")
        print(f"   Full training (50 epochs): {full_time_max/3600:.1f} hours")
        print(f"   With early stopping (~30 epochs): {full_time_est/3600:.1f} hours")
        print(f"   Recommended job time: {full_time_max*1.5/3600:.0f} hours")
        
        # Training performance
        if results and 'train_loss' in results:
            print(f"\n📈 PERFORMANCE (1 epoch):")
            print(f"   Final train loss: {results['train_loss']:.4f}")
            print(f"   Final val loss: {results['val_loss']:.4f}")
            print(f"   Samples processed: {total_samples:,}")
            print(f"   Samples/second: {total_samples/epoch_time:.0f}")
        
        # SLURM job recommendations
        job_hours = max(6, int(full_time_max*1.5/3600) + 1)  # At least 6 hours
        print(f"\n🖥️  SLURM JOB SETTINGS:")
        print(f"   Recommended time: {job_hours}:00:00")
        print(f"   Memory: 32GB (for full dataset)")
        print(f"   GPUs: 1 (if available)")
        print(f"   CPUs: 4-8")
        
    except Exception as e:
        end_time = time.time()
        epoch_time = end_time - start_time
        print(f"\n❌ TIMING TEST FAILED after {epoch_time:.1f} seconds")
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()