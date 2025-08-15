#!/bin/bash
#SBATCH --job-name=kfold_mae
#SBATCH --account=def-pfieguth
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:4

echo "🔄 Enhanced Biomass Training - 5-Fold Cross-Validation with MAE"
echo "=============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start Time: $(date)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 32GB"
echo "GPUs: 4x GPU"
echo "=============================================================="

# Load modules
module load python/3.11
source myenv311/bin/activate

# GPU info
nvidia-smi

echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"

echo "=============================================================="
echo "📊 K-Fold Training Configuration:"
echo "Loss function: MAE (Mean Absolute Error)"
echo "Cross-validation: 5-fold stratified K-fold"
echo "Filtering: Outlier filtering enabled + max 10mg biomass"
echo "Dataset: Full dataset after filtering"
echo "Batch size: 32"
echo "Epochs per fold: 80"
echo "Learning rate: 0.0001"
echo "Random seed: 42 (reproducible)"
echo "Output directory: enhanced_model_outputs/kfold_*"
echo "=============================================================="

# Check required files
if [ ! -f "bin_results/bin_id_biomass_mapping.csv" ]; then
    echo "❌ BIN mapping CSV not found!"
    exit 1
fi

if [ ! -f "image_index.json" ]; then
    echo "❌ Image index JSON not found!"
    exit 1
fi

echo "✅ Required files found. Starting K-fold training..."

# Run K-fold training
echo "🏋️ Starting 5-FOLD CROSS-VALIDATION with MAE loss..."
python fixed_training_pipeline.py \
    --bin_mapping_csv bin_results/bin_id_biomass_mapping.csv \
    --image_index_json image_index.json \
    --batch_size 32 \
    --num_epochs 80 \
    --learning_rate 0.0001 \
    --output_dir enhanced_model_outputs \
    --loss_function mae \
    --use_kfold \
    --k_folds 5 \
    --filter_outliers \
    --max_biomass 10.0 \
    --random_seed 42

echo "=============================================================="
echo "✅ K-Fold training completed at: $(date)"
echo "🔍 Check enhanced_model_outputs/kfold_*/ for results"
echo "📊 Each fold model and statistics saved separately"
echo "=============================================================="