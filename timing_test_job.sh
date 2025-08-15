#!/bin/bash
#SBATCH --job-name=timing_test
#SBATCH --account=def-pfieguth
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=timing_test-%j.out

echo "🕐 Starting Timing Test: 1 Epoch on Cleaned Dataset"
echo "================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 16GB"
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
    "timing_test.py"
)

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Required file not found: $file"
        exit 1
    fi
done
echo "✅ All required files found"

# Show dataset stats
echo "📊 Dataset Information:"
python -c "
import pandas as pd
df = pd.read_csv('bin_results/bin_id_biomass_mapping_cleaned.csv')
total_samples = df['specimen_count'].sum()
print(f'   BIN groups: {len(df):,}')
print(f'   Total samples: {total_samples:,}')
print(f'   Biomass range: {df[\"mean_weight\"].min():.2f} - {df[\"mean_weight\"].max():.2f} mg')
"

# Run timing test
echo "⏰ Starting 1-epoch timing test..."
echo "Goal: Estimate time needed for full training job"
echo "================================================="

python timing_test.py

echo "================================================="
echo "✅ Timing test completed at: $(date)"
echo "Check output above for time estimates and job recommendations"
echo "================================================="