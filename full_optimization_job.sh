#!/bin/bash
#SBATCH --job-name=optimize_biomass
#SBATCH --account=def-pfieguth
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Activate environment
source ~/myenv311/bin/activate

# Change to project directory
cd ~/projects/def-pfieguth/vikramre/bioscan-new/

echo "🚀 Step 1: Creating Image Index"
echo "=================================="
python create_image_index.py

echo ""
echo "🚀 Step 2: Running Optimized ML Training"
echo "========================================"
python optimized_ml_pipeline.py \
    --bin_mapping_csv "bin_results/bin_id_biomass_mapping.csv" \
    --image_index_json "image_index.json" \
    --batch_size 32 \
    --num_epochs 20 \
    --learning_rate 0.001 \
    --output_dir "optimized_model_outputs" \
    --max_samples 1000

echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="