#!/usr/bin/env python3
"""
Simple test script to verify BIOSCAN import works
Just imports the package and tries to access one sample
"""

import sys
import time
from datetime import datetime

print(f"🧪 BIOSCAN Import Test - {datetime.now()}")
print(f"Python version: {sys.version}")

# Test 1: Basic import
print("\n1️⃣ Testing basic import...")
try:
    start_time = time.time()
    from bioscan_dataset import BIOSCAN5M
    import_time = time.time() - start_time
    print(f"✅ Import successful in {import_time:.2f} seconds")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Try to create dataset object (no download)
print("\n2️⃣ Testing dataset creation (no download)...")
try:
    start_time = time.time()
    dataset = BIOSCAN5M(
        root='./bioscan_data',
        split='train',
        download=False,  # Don't download, just check existing
        modality=('image',)
    )
    creation_time = time.time() - start_time
    print(f"✅ Dataset creation successful in {creation_time:.2f} seconds")
    print(f"📊 Dataset size: {len(dataset)} samples")
except Exception as e:
    print(f"⚠️ Dataset creation failed (expected if no data): {e}")

# Test 3: Try minimal download (just check the mechanism)
print("\n3️⃣ Testing download mechanism...")
try:
    start_time = time.time()
    dataset = BIOSCAN5M(
        root='./bioscan_data',
        split='train',
        download=True,
        modality=('image',)
    )
    download_time = time.time() - start_time
    print(f"✅ Download mechanism works in {download_time:.2f} seconds")
    print(f"📊 Dataset size: {len(dataset)} samples")
    
    # Try to access first sample
    if len(dataset) > 0:
        print("\n4️⃣ Testing sample access...")
        sample = dataset[0]
        print(f"✅ First sample accessed successfully")
        print(f"📋 Sample keys: {list(sample.keys()) if isinstance(sample, dict) else 'Not a dict'}")
        
except Exception as e:
    print(f"❌ Download test failed: {e}")
    print("This could be due to network issues or hanging import")

print(f"\n🎯 Test completed - {datetime.now()}")