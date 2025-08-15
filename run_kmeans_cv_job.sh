#!/bin/bash
#SBATCH --job-name=kmeans_cv
#SBATCH --account=def-pfieguth
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=kmeans_cv-%j.out

echo "🎯 K-MEANS 5-FOLD CROSS-VALIDATION"
echo "=================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 32GB"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=================================="

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

echo ""
echo "🎯 K-MEANS CROSS-VALIDATION OVERVIEW:"
echo "   Purpose: Robust evaluation of biomass prediction model"
echo "   Method: 5-fold CV with k-means clustering on biomass values"
echo "   Benefits: Ensures balanced weight distribution across folds"
echo "   Dataset: Cleaned (11,234 samples, 592 outliers removed)"
echo "   Baseline: 0.52 mg MAE to beat"
echo "   Expected: More reliable performance estimate than single holdout"

# Check required files
echo ""
echo "📁 Checking required files..."
required_files=(
    "bin_results/bin_id_biomass_mapping_cleaned.csv"
    "image_index.json"
    "kmeans_cross_validation.py"
    "fixed_training_pipeline.py"
)

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Required file not found: $file"
        exit 1
    fi
done
echo "✅ All required files found"

# Run k-means cross-validation
echo ""
echo "🚀 Starting k-means 5-fold cross-validation..."
echo "=============================================="

python3 kmeans_cross_validation.py

echo ""
echo "=============================================="
echo "✅ K-means cross-validation completed at: $(date)"
echo "📊 Check results above for robust performance metrics"
echo "🎯 Results provide unbiased estimate of model generalization"
echo "=============================================="