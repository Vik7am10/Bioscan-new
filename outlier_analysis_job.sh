#!/bin/bash
#SBATCH --job-name=outlier_analysis
#SBATCH --account=def-pfieguth
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=outlier_analysis-%j.out

echo "🔍 Starting Biomass Outlier Analysis"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 16GB"
echo "=========================================="

# Change to submission directory
cd $SLURM_SUBMIT_DIR

# Show Python info
echo "Python version: $(python --version)"
echo "Working directory: $(pwd)"

# Check required files
echo "✅ Checking required files..."
if [ ! -f "bin_results/bin_id_biomass_mapping.csv" ]; then
    echo "❌ BIN mapping CSV not found!"
    exit 1
fi

if [ ! -f "image_index.json" ]; then
    echo "❌ Image index JSON not found!"
    exit 1
fi

echo "✅ Required files found"

# Check/install required packages
echo "🔍 Checking required packages..."
python -c "import cv2, sklearn, scipy, tqdm; print('✅ All packages available')" || {
    echo "❌ Missing packages. Installing..."
    pip install opencv-python scikit-learn scipy tqdm matplotlib
}

# Run outlier analysis
echo "🚀 Starting outlier analysis on 1000 samples..."
echo "Expected features to analyze:"
echo "  - Pixel area (foreground pixels)"
echo "  - Cube area (area^1.5 as volume proxy)"
echo "  - Bounding box area, contour area"
echo "  - Aspect ratio, solidity"
echo "Target: Identify samples with unrealistic biomass (>600mg)"
echo "=========================================="

python biomass_outlier_analysis.py

echo "✅ Outlier analysis completed!"
echo "📄 Output files generated:"
echo "  - biomass_outliers.csv (samples to potentially remove)"
echo "  - biomass_correlations.csv (feature correlations)"
echo "  - biomass_feature_analysis.csv (full analysis)"
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="