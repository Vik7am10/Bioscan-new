#!/bin/bash
#SBATCH --job-name=full_training_cleaned
#SBATCH --account=def-pfieguth
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=full_training_cleaned-%j.out

echo "🚀 FULL TRAINING: Cleaned Dataset (11,234 samples)"
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

# Check required files
echo "📁 Checking required files..."
required_files=(
    "bin_results/bin_id_biomass_mapping_cleaned.csv"
    "image_index.json"
    "fixed_training_pipeline.py"
)

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Required file not found: $file"
        exit 1
    fi
done
echo "✅ All required files found"

# Create models directory
mkdir -p models

# Show dataset stats
echo "📊 Cleaned Dataset Information:"
python -c "
import pandas as pd
df = pd.read_csv('bin_results/bin_id_biomass_mapping_cleaned.csv')
total_samples = df['specimen_count'].sum()
print(f'   BIN groups: {len(df):,}')
print(f'   Total samples: {total_samples:,}')
print(f'   Biomass range: {df[\"mean_weight\"].min():.2f} - {df[\"mean_weight\"].max():.2f} mg')
print(f'   Target baseline to beat: 0.52 mg MAE')
"

# Full training configuration
echo "🔧 Training Configuration:"
echo "   Dataset: CLEANED (592 outliers removed)"
echo "   Splits: 70% train, 15% validation, 15% TEST HOLDOUT"
echo "   Target: Beat 0.52 mg MAE baseline"
echo "   Epochs: 50 (with early stopping)"
echo "   Batch size: 32"
echo "   Learning rate: 0.0001"
echo "   Early stopping patience: 15"
echo "   Expected time: ~15 minutes (based on timing test)"

# Run full training
echo "🚀 Starting full training on cleaned dataset..."
echo "================================================="

python3 -c "
import sys
sys.path.append('.')
from fixed_training_pipeline import train_model

# Configuration for full training on cleaned dataset
config = {
    'bin_mapping_csv': 'bin_results/bin_id_biomass_mapping_cleaned.csv',
    'image_index_json': 'image_index.json',
    'max_samples': None,  # Use all 11,234 samples
    'batch_size': 32,
    'learning_rate': 0.0001,
    'num_epochs': 50,
    'patience': 15,  # Early stopping
    'val_split': 0.2,
    'device': 'auto',
    'save_model': True,
    'seed': 42,
    'use_cleaned': True,
    'output_dir': './models'
}

print('🎯 Goal: Beat 0.52 mg MAE baseline on cleaned dataset')
print('📈 Expected: Significant improvement with 11x more clean data')
print('⏰ Starting training...')

# Run training
results = train_model(config)

print()
print('✅ TRAINING COMPLETED!')
print('=' * 50)
if results:
    print(f'Final train loss: {results[\"train_loss\"]:.4f}')
    print(f'Final val loss: {results[\"val_loss\"]:.4f}')
    print(f'Best epoch: {results.get(\"best_epoch\", \"N/A\")}')
    
    # Convert losses back to MAE in mg for comparison
    # Note: losses are normalized, need to denormalize for real MAE
    print()
    print('🎯 Performance vs Baseline:')
    print('   Baseline to beat: 0.52 mg MAE')
    print('   Model performance: See validation metrics above')
    print('   Success metric: Val loss should be low and decreasing')
else:
    print('No results returned - check training logs above')
"

echo "================================================="
echo "✅ Full training completed at: $(date)"
echo "Check model performance above and in models/ directory"
echo "Next step: Evaluate final model against 0.52 mg MAE baseline"
echo "================================================="