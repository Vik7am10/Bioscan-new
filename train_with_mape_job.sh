#!/bin/bash
#SBATCH --job-name=train_mape_loss
#SBATCH --account=def-pfieguth
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=train_mape_loss-%j.out

echo "🎯 TRAINING WITH MAPE LOSS: Proper 70/15/15 Splits"
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
mkdir -p models_mape

echo "🎯 TRAINING CONFIGURATION WITH MAPE LOSS:"
echo "   Loss Function: Symmetric MAPE (intuitive percentage errors)"
echo "   Dataset: CLEANED (592 outliers removed)"
echo "   Splits: 70% train, 15% validation, 15% TEST HOLDOUT"
echo "   Benefits: Loss values = actual percentage errors"
echo "   Example: Loss=20.5 means 20.5% average error"

# Run training with MAPE loss
echo "🚀 Starting training with MAPE loss..."
echo "================================================="

python3 -c "
import sys
sys.path.append('.')
from fixed_training_pipeline import train_model

# Configuration for MAPE-based training
config = {
    'bin_mapping_csv': 'bin_results/bin_id_biomass_mapping_cleaned.csv',
    'image_index_json': 'image_index.json',
    'max_samples': None,  # Use all samples
    'batch_size': 32,
    'learning_rate': 0.0001,  # May need adjustment for MAPE
    'num_epochs': 50,
    'patience': 15,
    'val_split': 0.15,
    'device': 'auto',
    'save_model': True,
    'seed': 42,
    'use_cleaned': True,
    'output_dir': './models_mape',
    'loss_type': 'smape'  # Use Symmetric MAPE
}

print('🎯 Training with Symmetric MAPE Loss')
print('📊 Benefits:')
print('   - Loss values = percentage errors (interpretable)')
print('   - SMAPE is more robust than regular MAPE')
print('   - Values like Loss=15.3 mean 15.3% average error')
print('   - More intuitive than normalized losses')
print()

# Run training
results = train_model(config)

print()
print('✅ MAPE TRAINING COMPLETED!')
print('=' * 60)
if results:
    train_loss = results['train_loss']
    val_loss = results['val_loss']
    print(f'Final train SMAPE: {train_loss:.1f}%')
    print(f'Final val SMAPE: {val_loss:.1f}%')
    print(f'Best epoch: {results.get(\"best_epoch\", \"N/A\")}')
    
    print()
    print('🎯 Performance Interpretation:')
    if val_loss < 10:
        print(f'   EXCELLENT: {val_loss:.1f}% error is very good')
    elif val_loss < 20:
        print(f'   GOOD: {val_loss:.1f}% error is acceptable')
    elif val_loss < 50:
        print(f'   MODERATE: {val_loss:.1f}% error needs improvement')
    else:
        print(f'   POOR: {val_loss:.1f}% error is too high')
        
    print()
    print('📊 Expected baseline comparison:')
    print(f'   - Previous baseline: ~3000% MAPE (very poor)')
    print(f'   - This model: {val_loss:.1f}% MAPE')
    print(f'   - Improvement: {3000/val_loss:.1f}x better than baseline')
else:
    print('No results returned - check training logs above')
"

echo "================================================="
echo "✅ MAPE training completed at: $(date)"
echo "🎯 Model ready for TRUE holdout testing with interpretable metrics"
echo "📁 New model saved in models_mape/ directory"
echo "================================================="