#!/bin/bash
#SBATCH --job-name=enhanced_mae
#SBATCH --account=def-pfieguth
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:4

echo "🚀 Enhanced Biomass Training - MAE Loss with Outlier Filtering"
echo "============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start Time: $(date)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 32GB"
echo "GPUs: 4x GPU"
echo "============================================================="

# Load modules
module load python/3.11
source myenv311/bin/activate

# GPU info
nvidia-smi

echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"

echo "============================================================="
echo "📊 Enhanced Training Configuration:"
echo "Loss function: MAE (Mean Absolute Error)"
echo "Filtering: Outlier filtering enabled (scaling law)"
echo "Training mode: Single train/val split (no K-fold)"
echo "Dataset: Full dataset after outlier filtering (~11,234 samples)"
echo "Batch size: 32"
echo "Epochs: 100"
echo "Learning rate: 0.0001"
echo "Output directory: enhanced_model_outputs"
echo "Key features:"
echo "  - MAE loss for direct biomass optimization"
echo "  - Automatic scaling law outlier filtering"
echo "  - All samples after filtering (no max_samples limit)"
echo "  - Enhanced target normalization"
echo "  - Stratified train/val splits"
echo "============================================================="

# Check required files
if [ ! -f "bin_results/bin_id_biomass_mapping.csv" ]; then
    echo "❌ BIN mapping CSV not found!"
    exit 1
fi

if [ ! -f "image_index.json" ]; then
    echo "❌ Image index JSON not found!"
    exit 1
fi

echo "✅ Required files found. Starting enhanced training..."

# Check if biomass analysis results exist
if [ ! -f "biomass_feature_analysis.csv" ]; then
    echo "⚠️  Warning: biomass_feature_analysis.csv not found"
    echo "   Run biomass_outlier_analysis.py first for best results"
fi

# Run enhanced training
echo "🏋️ Starting ENHANCED biomass training with MAE loss..."
python fixed_training_pipeline.py \
    --bin_mapping_csv bin_results/bin_id_biomass_mapping.csv \
    --image_index_json image_index.json \
    --batch_size 32 \
    --num_epochs 100 \
    --learning_rate 0.0001 \
    --output_dir enhanced_model_outputs \
    --loss_function mae \
    --filter_outliers \
    --random_seed 42

echo "============================================================="
echo "✅ Training completed at: $(date)"
echo "🔍 Check enhanced_model_outputs/ for results"
echo "============================================================="