#!/bin/bash
#SBATCH --account=def-pfieguth
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=bin_mapping_full
#SBATCH --output=bin_mapping_full_%j.out
#SBATCH --error=bin_mapping_full_%j.err

echo "==========================================
Job ID: $SLURM_JOB_ID
Job Name: bin_mapping_full
Node: $SLURM_JOB_NODELIST
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
echo "Expected dataset size: 5+ million records"
echo "=========================================="

# Check if biomass file exists
if [ ! -f "5m_mass_metadata_avg.csv" ]; then
    echo "❌ Biomass file not found: 5m_mass_metadata_avg.csv"
    exit 1
fi

echo "✅ Biomass file found"

# Check if metadata exists
METADATA_FILE="./bioscan_data/bioscan5m/metadata/csv/BIOSCAN_5M_Insect_Dataset_metadata.csv"
if [ ! -f "$METADATA_FILE" ]; then
    echo "❌ BIOSCAN metadata file not found: $METADATA_FILE"
    echo "Please run fix_bioscan_download.py first"
    exit 1
fi

# Check metadata file size
METADATA_SIZE=$(stat -c%s "$METADATA_FILE")
echo "📊 BIOSCAN metadata file size: $((METADATA_SIZE / 1024 / 1024)) MB"

if [ $METADATA_SIZE -lt 1000000 ]; then
    echo "❌ Metadata file is too small (< 1MB), likely corrupted"
    exit 1
fi

echo "✅ BIOSCAN metadata file looks valid"

# Test import first with timeout
echo "🔍 Testing bioscan-dataset import..."
timeout 60 python -c "
try:
    from bioscan_dataset import BIOSCAN5M
    print('✅ BIOSCAN5M import successful')
except Exception as e:
    print(f'❌ Import failed: {e}')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ BIOSCAN import test failed or timed out"
    exit 1
fi

echo "🚀 Starting BIN mapping analysis on large dataset..."
echo "⏳ This may take 10-20 minutes for 5M+ records..."

# Run the analysis with explicit paths
python create_bin_mapping.py \
    --biomass_file 5m_mass_metadata_avg.csv \
    --bioscan_root ./bioscan_data \
    --output_dir ./bin_results

# Check exit status
if [ $? -eq 0 ]; then
    echo "✅ BIN mapping analysis completed successfully"
    
    # Create timestamped archive
    timestamp=$(date +%Y%m%d_%H%M)
    tar -czf bin_results_full_${timestamp}.tar.gz bin_results/
    
    echo "📦 Results archived: bin_results_full_${timestamp}.tar.gz"
    echo "📊 Output files:"
    ls -la bin_results/
    
    # Show summary if available
    if [ -f "bin_results/data_summary_statistics.txt" ]; then
        echo "📈 Summary statistics:"
        cat bin_results/data_summary_statistics.txt
    fi
    
    # Show BIN mapping preview
    if [ -f "bin_results/bin_id_biomass_mapping.csv" ]; then
        echo "🧬 BIN mapping preview (first 10 rows):"
        head -10 bin_results/bin_id_biomass_mapping.csv
    fi
else
    echo "❌ BIN mapping analysis failed"
    echo "Check the error logs for details"
    exit 1
fi

echo "🎯 Job completed: $(date)"
