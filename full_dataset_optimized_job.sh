#!/bin/bash
#SBATCH --job-name=full_optimized_biomass
#SBATCH --account=def-pfieguth
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:4
#SBATCH --output=full_optimized_biomass_%j.out
#SBATCH --error=full_optimized_biomass_%j.err

# Load required modules
module load StdEnv/2023 scipy-stack/2025a

# Activate virtual environment
source ~/myenv311/bin/activate

# Navigate to project directory
cd /home/vikramre/projects/def-pfieguth/vikramre/bioscan-new/

# Print job information
echo "=========================================="
echo "🚀 Full Dataset Optimized Biomass Training"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start Time: $(date)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 32GB"
echo "GPUs: 4x GPU"
echo "=========================================="

# Print environment information
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"

# Check if GPU is available and print GPU info
if python -c 'import torch; print(torch.cuda.is_available())' | grep -q True; then
    echo "GPU Info:"
    nvidia-smi
fi

echo "=========================================="

# Set training parameters for FULL DATASET
BIN_MAPPING_CSV="bin_results/bin_id_biomass_mapping.csv"
IMAGE_INDEX_JSON="image_index.json"
BATCH_SIZE=128       # Increased batch size for 4x GPU
NUM_EPOCHS=150       # More epochs for full training
LEARNING_RATE=0.001
OUTPUT_DIR="full_optimized_model_outputs"
# NO MAX_SAMPLES - use full dataset

# Create output directory
mkdir -p $OUTPUT_DIR

echo "🎯 Full Dataset Optimized Training Configuration:"
echo "BIN mapping CSV: $BIN_MAPPING_CSV"
echo "Image index JSON: $IMAGE_INDEX_JSON"
echo "Batch size: $BATCH_SIZE (optimized for 4x GPU)"
echo "Epochs: $NUM_EPOCHS (full training)"
echo "Learning rate: $LEARNING_RATE"
echo "Output directory: $OUTPUT_DIR"
echo "Sample limit: NONE (using full ~20K samples)"
echo "Expected training samples: ~16,267 (80%)"
echo "Expected validation samples: ~4,067 (20%)"
echo "=========================================="

# Check if required files exist
if [ ! -f "$BIN_MAPPING_CSV" ]; then
    echo "❌ Error: BIN mapping CSV not found: $BIN_MAPPING_CSV"
    echo "Please ensure bin mapping results are available."
    exit 1
fi

if [ ! -f "$IMAGE_INDEX_JSON" ]; then
    echo "❌ Error: Image index JSON not found: $IMAGE_INDEX_JSON"
    echo "Please ensure image index is available."
    exit 1
fi

echo "✅ Required files found. Starting full dataset training..."

# Install required packages
echo "🔄 Installing required packages..."
pip install torchmetrics pillow

# Run the full dataset optimized training
echo "🚀 Starting FULL dataset optimized training..."
python optimized_ml_pipeline.py \
    --bin_mapping_csv "$BIN_MAPPING_CSV" \
    --image_index_json "$IMAGE_INDEX_JSON" \
    --batch_size "$BATCH_SIZE" \
    --num_epochs "$NUM_EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --output_dir "$OUTPUT_DIR"

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "✅ Full dataset optimized training completed successfully!"
    echo "Model outputs saved to: $OUTPUT_DIR"
    
    # List output files
    echo "Generated files:"
    ls -la "$OUTPUT_DIR"
    
    # Print training summary
    echo "=========================================="
    echo "📊 Full Dataset Training Summary:"
    echo "- Used full dataset (~20,334 samples)"
    echo "- Training samples: ~16,267"
    echo "- Validation samples: ~4,067" 
    echo "- Batch size: $BATCH_SIZE"
    echo "- Epochs: $NUM_EPOCHS"
    echo "- GPU acceleration: 4x GPU"
    echo "- No sample limits applied"
    echo "=========================================="
else
    echo "❌ Training failed with exit code: $?"
    exit 1
fi

echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="