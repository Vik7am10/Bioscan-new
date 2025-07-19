#!/bin/bash
#SBATCH --account=def-pfieguth
#SBATCH --time=6:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=bin_mapping
#SBATCH --output=bin_mapping_%j.out
#SBATCH --error=bin_mapping_%j.err

echo "==========================================
Job ID: $SLURM_JOB_ID
Job Name: bin_mapping
Node: $SLURMD_NODENAME
Start Time: $(date)
=========================================="

# Setup environment
module load StdEnv/2023 scipy-stack/2025a
source ~/myenv311/bin/activate

echo "Python version: $(python --version)"
echo "Environment: $(which python)"

echo "=========================================="
echo "🧬 BIN Mapping Analysis Configuration:"
echo "Biomass file: 5m_mass_metadata_avg.csv"
echo "BIOSCAN root: ./bioscan_data"
echo "Output directory: ./bin_results"
echo "Download enabled: Yes"
echo "=========================================="

# Check if biomass file exists
if [ ! -f "5m_mass_metadata_avg.csv" ]; then
    echo "❌ Biomass file not found: 5m_mass_metadata_avg.csv"
    exit 1
fi

echo "✅ Biomass file found"

# Test import first
echo "🔍 Testing bioscan-dataset import..."
python -c "
try:
    from bioscan_dataset import BIOSCAN5M
    print('✅ BIOSCAN5M import successful')
except ImportError as e:
    print(f'❌ Import failed: {e}')
    exit(1)
except Exception as e:
    print(f'⚠️ Import error: {e}')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ BIOSCAN import test failed"
    exit 1
fi

echo "🚀 Starting BIN mapping analysis..."

# Run the analysis
python create_bin_mapping.py \
    --biomass_file 5m_mass_metadata_avg.csv \
    --bioscan_root ./bioscan_data \
    --output_dir ./bin_results \
    --download

# Check exit status
if [ $? -eq 0 ]; then
    echo "✅ BIN mapping analysis completed successfully"
    
    # Create timestamped archive
    timestamp=$(date +%Y%m%d_%H%M)
    tar -czf bin_results_${timestamp}.tar.gz bin_results/
    
    echo "📦 Results archived: bin_results_${timestamp}.tar.gz"
    echo "📊 Output files:"
    ls -la bin_results/
    
    # Show summary if available
    if [ -f "bin_results/data_summary_statistics.txt" ]; then
        echo "📈 Summary statistics:"
        cat bin_results/data_summary_statistics.txt
    fi
else
    echo "❌ BIN mapping analysis failed"
    echo "Check the error logs for details"
    exit 1
fi

echo "🎯 Job completed: $(date)"