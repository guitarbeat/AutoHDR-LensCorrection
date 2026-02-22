import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from typing import Optional, Tuple

class AutoHDRDataset(Dataset):
    """
    A custom PyTorch Dataset for the Kaggle Automatic Lens Correction competition.
    
    Expected Directory Structure:
    root_dir/
      train/
         pair_0/
           original.jpg
           generated.jpg
         pair_1/
           original.jpg
           generated.jpg
         ...
      test/
         img_0.jpg
         img_1.jpg
         ...
    """
    def __init__(self, root_dir: str, mode: str = "train", transform: Optional[transforms.Compose] = None, limit: Optional[int] = None):
        """
        Args:
            root_dir (str): Path to the extracted Kaggle dataset (e.g., /Volumes/Love SSD/automatic-lens-correction)
            mode (str): "train", "val", or "test"
            transform (callable, optional): Optional transform to be applied on an image.
            limit (int, optional): Limit the number of samples loaded (useful for quick testing/micro-training).
        """
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        
        self.samples = []
        
        if mode in ["train", "val"]:
            train_dir = os.path.join(root_dir, "train")
            if os.path.exists(train_dir):
                # Using os.scandir for better performance over os.listdir + os.path.isdir
                pair_dirs = []
                with os.scandir(train_dir) as it:
                    for entry in it:
                        if entry.is_dir():
                            pair_dirs.append(entry.name)
                            if limit is not None and len(pair_dirs) >= limit:
                                break

                # Using a sorted list of directories to ensure deterministic ordering
                # for train/val splits
                pair_dirs.sort()
                
                # Simple split: 95% train, 5% val
                split_idx = int(len(pair_dirs) * 0.95)
                
                if mode == "train":
                    target_dirs = pair_dirs[:split_idx]
                else:
                    target_dirs = pair_dirs[split_idx:]
                    
                for d in target_dirs:
                    pair_path = os.path.join(train_dir, d)
                    self.samples.append({
                        "original": os.path.join(pair_path, "original.jpg"),
                        "generated": os.path.join(pair_path, "generated.jpg")
                    })
        elif mode == "test":
            test_dir = os.path.join(root_dir, "test")
            if os.path.exists(test_dir):
                filenames = []
                with os.scandir(test_dir) as it:
                    for entry in it:
                        if entry.is_file() and entry.name.endswith(".jpg"):
                            filenames.append(entry.name)
                            if limit is not None and len(filenames) >= limit:
                                break
                filenames.sort()

                for f in filenames:
                    self.samples.append({
                        "original": os.path.join(test_dir, f),
                        "image_id": f.replace(".jpg", "")
                    })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample_info = self.samples[idx]
        
        original_img = Image.open(sample_info["original"]).convert("RGB")
        
        if self.transform:
            original_img = self.transform(original_img)
            
        result = {"original": original_img}
        
        if self.mode in ["train", "val"]:
            generated_img = Image.open(sample_info["generated"]).convert("RGB")
            if self.transform:
                generated_img = self.transform(generated_img)
            result["generated"] = generated_img
        else:
            result["image_id"] = sample_info["image_id"]
            
        return result

def get_dataloaders(root_dir: str, batch_size: int = 16, num_workers: int = 4, limit: Optional[int] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Helper function to build train, val, and test DataLoaders.
    ViT models usually expect 224x224 input.
    """
    vit_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Standard ImageNet normalization used by pretrained ViT
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = AutoHDRDataset(root_dir, mode="train", transform=vit_transform, limit=limit)
    val_dataset = AutoHDRDataset(root_dir, mode="val", transform=vit_transform, limit=limit)
    test_dataset = AutoHDRDataset(root_dir, mode="test", transform=vit_transform, limit=limit)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader
