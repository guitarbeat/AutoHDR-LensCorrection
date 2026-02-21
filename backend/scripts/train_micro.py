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

def train_micro_dataset(data_dir: str, num_samples: int = 20, epochs: int = 100):
    """
    Overfitting test: Trains the Vision Transformer on a tiny subset of the data.
    If the model architecture and loss function are correct, the loss should 
    approach zero rapidly.
    """
    print(f"=== Starting Micro-Dataset Overfitting Test ({num_samples} samples) ===")
    
    # 1. Setup Hardware
    system_hardware.print_system_info()
    device = system_hardware.get_tensor_device()
    
    # 2. Setup Data
    print(f"Loading data from: {data_dir}")
    train_loader, _, _ = get_dataloaders(root_dir=data_dir, batch_size=4, num_workers=0)
    
    # Extract a micro-subset
    if len(train_loader.dataset) == 0:
        print("Error: No training data found! Please ensure the Kaggle dataset is unzipped.")
        return
        
    micro_dataset = Subset(train_loader.dataset, range(min(num_samples, len(train_loader.dataset))))
    micro_loader = torch.utils.data.DataLoader(micro_dataset, batch_size=4, shuffle=True)
    
    # 3. Setup Model
    model = load_model(pretrained=True)
    model = model.to(device)
    model.train()
    
    # 4. Setup Loss & Optimizer
    # Kaggle evaluation relies heavily on Mean Absolute Error (L1 Loss)
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    # 5. Training Loop
    print("\nTraining...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for batch in micro_loader:
            original_imgs = batch["original"].to(device)
            # Flow-Field Approach:
            # The ViT outputs a (B, 2, 14, 14) grid of displacement vectors.
            # In a full training pipeline, we would compute the ground truth flow field
            # between `original_imgs` and `batch["generated"]` using an optical flow algorithm
            # (like RAFT) offline, and use that as the target.
            # 
            # Alternatively, we can use `torch.nn.functional.grid_sample` to warp
            # `original_imgs` using the predicted flow, and calculate MAE against `generated_imgs`.
            
            # For this MVP overfit test, we mock a target flow field of zeros 
            # (assuming perfect input) to verify gradient flow through the ViT backbone.
            mock_target_flow = torch.zeros(original_imgs.size(0), 2, 14, 14).to(device)
            
            optimizer.zero_grad()
            predictions = model(original_imgs) # Output: (B, 2, 14, 14)
            
            loss = criterion(predictions, mock_target_flow)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(micro_loader)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Loss (MAE): {avg_loss:.6f}")
            
    print("=== Micro-Dataset Test Complete ===")
    print("If the final loss is near 0.0000, gradient flow is working!")

if __name__ == "__main__":
    # Expect the dataset to be unzipped at /Volumes/Love SSD/automatic-lens-correction
    # Provide the path via command line, or default to the SSD
    default_path = "/Volumes/Love SSD/automatic-lens-correction"
    data_path = sys.argv[1] if len(sys.argv) > 1 else default_path
    
    train_micro_dataset(data_path)
