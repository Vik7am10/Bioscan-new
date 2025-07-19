#!/bin/bash
#SBATCH --account=def-pfieguth
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=bin_mapping
#SBATCH --output=bin_mapping_%j.out
#SBATCH --error=bin_mapping_%j.err

echo "==========================================
Job ID: $SLURM_JOB_ID
Job Name: bin_mapping
Node: $SLURM_JOB_NODELIST
Start Time: $(date)
=========================================="

# Setup environment
module load StdEnv/2023 scipy-stack/2025a
source ~/myenv311/bin/activate

echo "Python version: $(python --version)"
echo "Environment: $(which python)"

echo "=========================================="
echo "🧬 Direct BIN Mapping Analysis"
echo "Using pandas-only approach (no bioscan-dataset package)"
echo "=========================================="

# Navigate to working directory
cd ~/projects/def-pfieguth/vikramre/bioscan-new/

# Check input files
echo "📁 Checking input files..."
echo "Biomass data:"
ls -lh 5m_mass_metadata_avg.csv 2>/dev/null || echo "❌ Biomass file not found"

echo "BIOSCAN metadata:"
ls -lh ~/scratch/bioscan_data/bioscan5m/metadata/csv/BIOSCAN_5M_Insect_Dataset_metadata.csv 2>/dev/null || echo "❌ BIOSCAN metadata not found"

# Update paths in script for cluster environment
echo "🔧 Updating file paths for cluster..."
sed -i 's|metadata_file = "./bioscan_data/|metadata_file = "~/scratch/bioscan_data/|g' direct_bin_mapping.py

# Check memory usage before starting
echo "💾 Current memory usage:"
free -h

echo "🚀 Starting BIN mapping analysis..."
python direct_bin_mapping.py

PYTHON_EXIT_CODE=$?

echo "💾 Memory usage after processing:"
free -h

if [ $PYTHON_EXIT_CODE -eq 0 ]; then
    echo "✅ BIN mapping completed successfully!"
    
    echo "📊 Results summary:"
    ls -lh bin_results/ 2>/dev/null || echo "No results directory found"
    
    if [ -f bin_results/summary_statistics.txt ]; then
        echo "📋 Analysis summary:"
        cat bin_results/summary_statistics.txt
    fi
    
    if [ -f bin_results/bin_id_biomass_mapping.csv ]; then
        echo "📈 BIN mapping file created:"
        echo "Rows: $(wc -l < bin_results/bin_id_biomass_mapping.csv)"
        echo "Columns: $(head -1 bin_results/bin_id_biomass_mapping.csv | tr ',' '\n' | wc -l)"
    fi
else
    echo "❌ BIN mapping failed with exit code: $PYTHON_EXIT_CODE"
    exit 1
fi

echo "🎯 Job completed: $(date)"