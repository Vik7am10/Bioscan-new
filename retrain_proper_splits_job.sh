#!/bin/bash
#SBATCH --job-name=retrain_proper_splits
#SBATCH --account=def-pfieguth
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=retrain_proper_splits-%j.out

echo "🔄 RETRAINING: Proper 70/15/15 Train/Val/Test Splits"
echo "================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 32GB"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "================================================="

# Change to submission directory
cd $SLURM_SUBMIT_DIR

# Activate virtual environment
echo "🐍 Activating virtual environment..."
source myenv311/bin/activate

# Show system info
echo "📊 System Information:"
echo "Python version: $(python --version)"
echo "Working directory: $(pwd)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    echo "GPU device: $(python -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo 'GPU info not available')"
fi

# Create models directory
mkdir -p models_proper_splits

# Show the data split breakdown
echo "📊 NEW Data Split Strategy:"
python -c "
import pandas as pd
df = pd.read_csv('bin_results/bin_id_biomass_mapping_cleaned.csv')
total_samples = df['specimen_count'].sum()
train_samples = int(0.7 * total_samples)
val_samples = int(0.15 * total_samples)
test_samples = total_samples - train_samples - val_samples
print(f'   Total samples: {total_samples:,}')
print(f'   Train (70%): {train_samples:,} samples')
print(f'   Validation (15%): {val_samples:,} samples')
print(f'   TEST HOLDOUT (15%): {test_samples:,} samples')
print(f'   Previous model saw validation set - NOW FIXED')
print(f'   Test set is TRUE HOLDOUT (never seen by model)')
"

echo "🔧 Training Configuration:"
echo "   Dataset: CLEANED (592 outliers removed)"
echo "   Splits: 70% train, 15% validation, 15% TEST HOLDOUT"
echo "   Target: Beat 0.52 mg MAE baseline on TRUE holdout test"
echo "   Epochs: 50 (with early stopping)"
echo "   Batch size: 32"
echo "   Learning rate: 0.0001"
echo "   Early stopping patience: 15"

# Run training with proper splits
echo "🚀 Starting training with proper data splits..."
echo "================================================="

python3 -c "
import sys
sys.path.append('.')
from fixed_training_pipeline import train_model

# Configuration for training with proper splits
config = {
    'bin_mapping_csv': 'bin_results/bin_id_biomass_mapping_cleaned.csv',
    'image_index_json': 'image_index.json',
    'max_samples': None,  # Use all samples
    'batch_size': 32,
    'learning_rate': 0.0001,
    'num_epochs': 50,
    'patience': 15,  # Early stopping
    'val_split': 0.15,  # 15% validation (out of 85% train+val)
    'device': 'auto',
    'save_model': True,
    'seed': 42,
    'use_cleaned': True,
    'output_dir': './models_proper_splits'
}

print('🎯 Goal: Train model that generalizes to TRUE holdout test set')
print('📊 Data splits: 70% train, 15% val (for early stopping), 15% test (never seen)')
print('⚠️  Previous model was overfitted to validation set')
print('✅ This model will be properly evaluated on unseen test data')
print()

# Run training
results = train_model(config)

print()
print('✅ TRAINING WITH PROPER SPLITS COMPLETED!')
print('=' * 60)
if results:
    print(f'Final train loss: {results[\"train_loss\"]:.4f}')
    print(f'Final val loss: {results[\"val_loss\"]:.4f}')
    print(f'Best epoch: {results.get(\"best_epoch\", \"N/A\")}')
    
    print()
    print('🎯 Next Steps:')
    print('   1. Test on TRUE holdout test set (15% never seen)')
    print('   2. Get realistic performance vs 0.52 mg baseline')
    print('   3. Previous 0.091 mg was overfitted - expect higher but fair MAE')
else:
    print('No results returned - check training logs above')
"

echo "================================================="
echo "✅ Training with proper splits completed at: $(date)"
echo "🎯 Model ready for TRUE holdout testing (no data leakage)"
echo "📁 New model saved in models_proper_splits/ directory"
echo "================================================="