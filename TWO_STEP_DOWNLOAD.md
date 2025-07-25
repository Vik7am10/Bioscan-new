# Two-Step BIOSCAN Download Solution

## Problem
Compute nodes on Beluga don't have internet access, causing `[Errno 101] Network is unreachable` errors when trying to download BIOSCAN-5M dataset.

## Solution
Use a two-step approach:
1. **Login Node**: Download dataset (has internet access)
2. **Compute Node**: Process dataset (has compute resources)

## Step 1: Download on Login Node

**Upload the download script:**
```bash
scp login_node_download.py beluga:~/projects/def-pfieguth/vikramre/bioscan-new/
```

**Run on login node (interactive):**
```bash
ssh beluga
cd ~/projects/def-pfieguth/vikramre/bioscan-new/
module load StdEnv/2023 scipy-stack/2025a
source ~/myenv311/bin/activate
python login_node_download.py
```

**Expected output:**
```
 BIOSCAN-5M Dataset Download (Login Node)
==========================================
 Available disk space: 25.4 GB
 Setting up environment...
 bioscan-dataset package available
 Downloading train split...
 Attempt 1/3 for train split...
 Successfully downloaded train split!
Dataset size: 3,505,971 samples
 Disk usage after train: 12.3GB
...
 BIOSCAN dataset download completed successfully!
```

## Step 2: Process on Compute Node

**Upload the compute job:**
```bash
scp compute_node_job.sh beluga:~/projects/def-pfieguth/vikramre/bioscan-new/
```

**Submit compute job:**
```bash
sbatch compute_node_job.sh
```

**Monitor progress:**
```bash
squeue -u vikramre
tail -f bioscan_compute_*.out
```

## Files Created

### `login_node_download.py`
- Downloads BIOSCAN-5M dataset on login node
- Robust error handling with retries
- Disk space monitoring
- Verification of downloaded data

### `compute_node_job.sh`
- SLURM job for compute node processing
- Uses pre-downloaded dataset
- Runs BIN mapping analysis
- No internet access required

## Advantages

1. **Reliable**: Separates download (internet) from processing (compute)
2. **Robust**: Retry logic handles temporary network issues
3. **Efficient**: Uses appropriate resources for each task
4. **Flexible**: Can process data multiple times without re-downloading

## Disk Space Requirements

- **BIOSCAN-5M dataset**: ~15-20GB
- **Available space**: Check with `df -h` before starting
- **Cleanup**: Remove old files if needed

## Troubleshooting

### Login Node Issues
- **Out of space**: Clean up old files or use different directory
- **Network timeout**: Script includes retry logic
- **Permission errors**: Ensure write access to target directory

### Compute Node Issues
- **Dataset not found**: Verify Step 1 completed successfully
- **Import errors**: Check that environment modules are loaded
- **Processing failures**: Check job logs for specific errors

## Next Steps After Success

1. **Verify dataset**: Check that all splits downloaded correctly
2. **Run BIN mapping**: Use `direct_bin_mapping.py` approach
3. **ML training**: Submit `hybrid_training_job.sh`
4. **Results analysis**: Download and analyze outputs

## Recovery Instructions

If the download fails:
1. Check available disk space
2. Clean up partial downloads: `rm -rf bioscan_data/`
3. Re-run login node download
4. Check network connectivity on login node

If processing fails:
1. Verify dataset integrity
2. Check compute node logs
3. Re-submit compute job
4. Use backup `direct_bin_mapping.py` if needed

---

**This approach bypasses the compute node network restrictions while maintaining the robust error handling needed for large dataset downloads.**