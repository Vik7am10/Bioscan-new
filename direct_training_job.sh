#!/bin/bash
#SBATCH --account=def-pfieguth
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name=direct_biomass_ml
#SBATCH --output=direct_ml_%j.out
#SBATCH --error=direct_ml_%j.err

echo "==========================================
Job ID: $SLURM_JOB_ID
Job Name: direct_biomass_ml
Node: $SLURM_JOB_NODELIST
Start Time: $(date)
=========================================="

# Setup environment
module load StdEnv/2023 scipy-stack/2025a
source ~/myenv311/bin/activate

echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU device: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\")')"

echo "=========================================="
echo "🚀 Direct Biomass ML Training Pipeline"
echo "Using BIN mapping results and direct image access"
echo "=========================================="

# Navigate to working directory
cd ~/projects/def-pfieguth/vikramre/bioscan-new/

# Check input files
echo "📁 Checking input files..."
echo "BIN mapping CSV:"
ls -lh bin_results/bin_id_biomass_mapping.csv 2>/dev/null || echo "❌ BIN mapping not found"

echo "BIOSCAN images:"
ls -ld ~/scratch/bioscan_data/bioscan5m/images/cropped_256/ 2>/dev/null || echo "❌ BIOSCAN images not found"

# Install required packages
echo "🔧 Installing/updating packages..."
pip install torchvision pillow torchmetrics

# Run training for each model type
echo "🎯 Training ResNet model..."
python direct_ml_pipeline.py \
    --bin_mapping_csv bin_results/bin_id_biomass_mapping.csv \
    --bioscan_images_root ~/scratch/bioscan_data/bioscan5m/images \
    --model_type resnet \
    --batch_size 32 \
    --num_epochs 25 \
    --learning_rate 0.001 \
    --output_dir ./direct_model_outputs \
    --max_samples 5000

RESNET_EXIT_CODE=$?

echo "🎯 Training Basic CNN model..."
python direct_ml_pipeline.py \
    --bin_mapping_csv bin_results/bin_id_biomass_mapping.csv \
    --bioscan_images_root ~/scratch/bioscan_data/bioscan5m/images \
    --model_type basic \
    --batch_size 32 \
    --num_epochs 25 \
    --learning_rate 0.001 \
    --output_dir ./direct_model_outputs \
    --max_samples 5000

BASIC_EXIT_CODE=$?

echo "🎯 Training Scale-enhanced CNN model..."
python direct_ml_pipeline.py \
    --bin_mapping_csv bin_results/bin_id_biomass_mapping.csv \
    --bioscan_images_root ~/scratch/bioscan_data/bioscan5m/images \
    --model_type scale \
    --batch_size 32 \
    --num_epochs 25 \
    --learning_rate 0.001 \
    --output_dir ./direct_model_outputs \
    --max_samples 5000

SCALE_EXIT_CODE=$?

# Check results
echo "📊 Training Results Summary:"
echo "ResNet model: $([[ $RESNET_EXIT_CODE -eq 0 ]] && echo 'SUCCESS' || echo 'FAILED')"
echo "Basic CNN model: $([[ $BASIC_EXIT_CODE -eq 0 ]] && echo 'SUCCESS' || echo 'FAILED')"
echo "Scale CNN model: $([[ $SCALE_EXIT_CODE -eq 0 ]] && echo 'SUCCESS' || echo 'FAILED')"

if [ -d direct_model_outputs ]; then
    echo "📁 Model outputs:"
    ls -lh direct_model_outputs/
else
    echo "❌ No model outputs directory found"
fi

echo "💾 Final memory usage:"
free -h

echo "🎯 Job completed: $(date)"