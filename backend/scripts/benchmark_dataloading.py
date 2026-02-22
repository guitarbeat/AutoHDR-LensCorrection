import os
import time
import shutil
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.dataloader import get_dataloaders

def create_dummy_dataset(root, num_samples=2000):
    if os.path.exists(root):
        shutil.rmtree(root)

    train_dir = os.path.join(root, "train")
    os.makedirs(train_dir, exist_ok=True)

    print(f"Creating {num_samples} dummy pairs in {train_dir}...")
    for i in range(num_samples):
        d = os.path.join(train_dir, f"pair_{i}")
        os.makedirs(d, exist_ok=True)
        # Touch files
        Path(os.path.join(d, "original.jpg")).touch()
        Path(os.path.join(d, "generated.jpg")).touch()

    test_dir = os.path.join(root, "test")
    os.makedirs(test_dir, exist_ok=True)
    # create some test files
    for i in range(10):
        Path(os.path.join(test_dir, f"img_{i}.jpg")).touch()

def benchmark():
    root = "dummy_dataset_bench"
    create_dummy_dataset(root, num_samples=20000)

    print("Benchmarking get_dataloaders (limit=None)...")
    start_time = time.time()
    try:
        get_dataloaders(root_dir=root, batch_size=4, num_workers=0, limit=None)
    except Exception as e:
        print(f"Error during benchmark (limit=None): {e}")
    end_time = time.time()
    print(f"Time taken (limit=None): {end_time - start_time:.6f} seconds")

    print("Benchmarking get_dataloaders (limit=100)...")
    start_time = time.time()
    try:
        get_dataloaders(root_dir=root, batch_size=4, num_workers=0, limit=100)
    except Exception as e:
        print(f"Error during benchmark (limit=100): {e}")
    end_time = time.time()
    print(f"Time taken (limit=100): {end_time - start_time:.6f} seconds")

    # Clean up
    shutil.rmtree(root)

if __name__ == "__main__":
    benchmark()
