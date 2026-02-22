import os
import shutil
import pytest
from pathlib import Path
import sys

# Ensure backend can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.core.dataloader import get_dataloaders, AutoHDRDataset

@pytest.fixture
def dummy_dataset(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    train_dir = root / "train"
    train_dir.mkdir()

    # Create 50 samples
    for i in range(50):
        d = train_dir / f"pair_{i}"
        d.mkdir()
        (d / "original.jpg").touch()
        (d / "generated.jpg").touch()

    test_dir = root / "test"
    test_dir.mkdir()
    for i in range(10):
        (test_dir / f"img_{i}.jpg").touch()

    return str(root)

def test_autohdr_dataset_limit(dummy_dataset):
    # Expect failure until implemented
    try:
        ds = AutoHDRDataset(dummy_dataset, mode="train", limit=10)
        # If limit works, we expect roughly 10 samples (minus validation split)
        # 10 * 0.95 = 9.5 -> 9 samples
        assert len(ds) == 9
    except TypeError:
        pytest.fail("AutoHDRDataset does not accept limit argument yet")

def test_get_dataloaders_limit(dummy_dataset):
    # Expect failure until implemented
    try:
        train_loader, val_loader, test_loader = get_dataloaders(dummy_dataset, batch_size=2, num_workers=0, limit=20)
        # Train split is 95% of 20 = 19
        assert len(train_loader.dataset) == 19
    except TypeError:
        pytest.fail("get_dataloaders does not accept limit argument yet")
