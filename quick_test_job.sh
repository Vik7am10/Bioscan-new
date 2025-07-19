#!/bin/bash
#SBATCH --job-name=quick_ml_test
#SBATCH --account=def-pfieguth
#SBATCH --time=0:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

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

# Run quick test
python quick_ml_test.py

echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="