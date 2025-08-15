#!/bin/bash
#SBATCH --job-name=eval_enhanced
#SBATCH --account=def-pfieguth
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

echo "🔍 Enhanced Model Evaluation"
echo "============================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start Time: $(date)"
echo "============================"

# Load modules
module load python/3.11
source myenv311/bin/activate

# GPU info
nvidia-smi

echo "============================"
echo "📊 Evaluation Configuration:"
echo "Model: Latest enhanced model"
echo "Test split: test"
echo "Comparison: Enhanced vs Baselines"
echo "Metrics: MAE, RMSE, MAPE, SMAPE, R²"
echo "Plots: Predictions, Residuals, Distributions"
echo "============================"

# Find the latest model
MODEL_PATH=$(find enhanced_model_outputs -name "*.pth" -type f -exec ls -t {} + | head -1)

if [ -z "$MODEL_PATH" ]; then
    echo "❌ No enhanced model found in enhanced_model_outputs/"
    echo "   Run enhanced training first"
    exit 1
fi

echo "🎯 Using model: $MODEL_PATH"

# Run evaluation
python evaluate_enhanced_model.py \
    --model_path "$MODEL_PATH" \
    --bin_mapping_csv bin_results/bin_id_biomass_mapping.csv \
    --image_index_json image_index.json \
    --output_dir evaluation_results \
    --test_split test

echo "============================"
echo "✅ Evaluation completed at: $(date)"
echo "📁 Check evaluation_results/ for plots and metrics"
echo "============================"