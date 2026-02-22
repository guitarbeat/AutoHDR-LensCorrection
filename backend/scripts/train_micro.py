import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset
import sys

# Add the project root to python path so we can import models and core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.detector import load_model
from core.hardware import system_hardware
from core.dataloader import get_dataloaders


def setup_training(data_dir: str, num_samples: int):
    """
    Sets up the training environment: hardware, data, model, and optimizer.
    """
    # 1. Setup Hardware
    system_hardware.print_system_info()
    device = system_hardware.get_tensor_device()

    # 2. Setup Data
    print(f"Loading data from: {data_dir}")
    train_loader, _, _ = get_dataloaders(root_dir=data_dir, batch_size=4, num_workers=0)

    # Extract a micro-subset
    if len(train_loader.dataset) == 0:
        raise ValueError(
            "No training data found! Please ensure the Kaggle dataset is unzipped."
        )

    micro_dataset = Subset(
        train_loader.dataset, range(min(num_samples, len(train_loader.dataset)))
    )
    micro_loader = torch.utils.data.DataLoader(
        micro_dataset, batch_size=4, shuffle=True
    )

    # 3. Setup Model
    # Note: load_model() does not accept arguments in its current implementation
    model = load_model()
    model = model.to(device)
    model.train()

    # 4. Setup Loss & Optimizer
    # Kaggle evaluation relies heavily on Mean Absolute Error (L1 Loss)
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    return model, micro_loader, criterion, optimizer, device


def run_training_loop(model, loader, criterion, optimizer, device, epochs: int):
    """
    Runs the training loop for the specified number of epochs.
    """
    print("\nTraining...")
    for epoch in range(epochs):
        epoch_loss = 0.0

        for batch in loader:
            original_imgs = batch["original"].to(device)
            # Distortion Coefficient Approach:
            # The ViT outputs a (B, 5) tensor of distortion coefficients.
            # In a full training pipeline, we would minimize the reconstruction error
            # or compare against known ground-truth coefficients if available.

            # For this MVP overfit test, we mock a target vector of zeros
            # (assuming perfect input) to verify gradient flow through the ViT backbone.
            mock_target_coeffs = torch.zeros(original_imgs.size(0), 5).to(device)

            optimizer.zero_grad()
            predictions = model(original_imgs)  # Output: (B, 5)

            loss = criterion(predictions, mock_target_coeffs)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Loss (MAE): {avg_loss:.6f}")


def train_micro_dataset(data_dir: str, num_samples: int = 20, epochs: int = 100):
    """
    Overfitting test: Trains the Vision Transformer on a tiny subset of the data.
    If the model architecture and loss function are correct, the loss should
    approach zero rapidly.
    """
    print(f"=== Starting Micro-Dataset Overfitting Test ({num_samples} samples) ===")

    try:
        model, micro_loader, criterion, optimizer, device = setup_training(
            data_dir, num_samples
        )
    except ValueError as e:
        print(f"Error: {e}")
        return

    run_training_loop(model, micro_loader, criterion, optimizer, device, epochs)

    print("=== Micro-Dataset Test Complete ===")
    print("If the final loss is near 0.0000, gradient flow is working!")


if __name__ == "__main__":
    # Expect the dataset to be unzipped at /Volumes/Love SSD/automatic-lens-correction
    # Provide the path via command line, or default to the SSD
    default_path = "/Volumes/Love SSD/automatic-lens-correction"
    data_path = sys.argv[1] if len(sys.argv) > 1 else default_path

    train_micro_dataset(data_path)
