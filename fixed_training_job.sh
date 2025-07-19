#!/bin/bash
#SBATCH --job-name=fixed_biomass
#SBATCH --account=def-pfieguth
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:4

echo "🚀 Fixed Biomass Training"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start Time: $(date)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 32GB"
echo "GPUs: 4x GPU"
echo "=========================================="

# Load modules
module load python/3.11
source ~/myenv311/bin/activate

# GPU info
nvidia-smi

echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"

echo "=========================================="
echo "🎯 Fixed Training Configuration:"
echo "BIN mapping CSV: bin_results/bin_id_biomass_mapping.csv"
echo "Image index JSON: image_index.json"
echo "Batch size: 32"
echo "Epochs: 50"
echo "Learning rate: 0.0001 (reduced from 0.001)"
echo "Output directory: fixed_model_outputs"
echo "Max samples: 1000 (for testing)"
echo "Key improvements:"
echo "  - Target normalization (mean/std scaling)"
echo "  - Lower learning rate with scheduler"
echo "  - Huber loss (robust to outliers)"
echo "  - Pretrained ResNet18 backbone"
echo "  - Dropout and regularization"
echo "  - Early stopping with patience=15"
echo "=========================================="

# Check required files
if [ ! -f "bin_results/bin_id_biomass_mapping.csv" ]; then
    echo "❌ BIN mapping CSV not found!"
    exit 1
fi

if [ ! -f "image_index.json" ]; then
    echo "❌ Image index JSON not found!"
    exit 1
fi

echo "✅ Required files found. Starting fixed training..."

# Install any missing packages
echo "🔄 Installing required packages..."
pip install torchmetrics pillow

# Run fixed training
echo "🚀 Starting FIXED biomass training..."
python fixed_training_pipeline.py \
    --bin_mapping_csv bin_results/bin_id_biomass_mapping.csv \
    --image_index_json image_index.json \
    --batch_size 32 \
    --num_epochs 50 \
    --learning_rate 0.0001 \
    --output_dir fixed_model_outputs \
    --max_samples 1000

echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="