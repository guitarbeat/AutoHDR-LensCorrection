import torch
import os

class HardwareManager:
    """
    Manages hardware resource allocation specifically designed for prototyping on
    Apple Silicon (M4) and scaling to GCP/Kaggle GPUs (CUDA).
    """
    def __init__(self):
        self.gpu_device = self._detect_gpu()
        self.cpu_device = torch.device('cpu')
        
    def _detect_gpu(self):
        """
        Detects the best available tensor accelerator.
        Prefers MPS for Apple Silicon prototyping, falls back to CUDA for GCP/Kaggle,
        then CPU if no hardware accelerators are available.
        """
        if torch.backends.mps.is_available():
            print("Accelerated hardware found: Apple Silicon MPS")
            return torch.device('mps')
        elif torch.cuda.is_available():
            print(f"Accelerated hardware found: CUDA ({torch.cuda.get_device_name(0)})")
            return torch.device('cuda')
        else:
            print("No accelerated hardware found. Falling back to CPU for tensor operations.")
            return torch.device('cpu')
            
    def get_tensor_device(self):
        """Returns the device designated for heavy deep learning inference (ViT)."""
        return self.gpu_device
        
    def get_io_device(self):
        """Returns the device designated for lightweight parsing and pre-processing."""
        return self.cpu_device

    def print_system_info(self):
        """Logs the hardware routing setup."""
        print("=== AutoHDR Hardware Routing ===")
        print(f"Tensor Compute Device: {self.gpu_device}")
        print(f"I/O Compute Device: {self.cpu_device}")
        
        # Thread tuning
        cpu_count = os.cpu_count() or 4
        print(f"Available Logical Cores: {cpu_count}")
        
        # We can explicitly set PyTorch thread limits for CPU fallback here
        torch.set_num_threads(cpu_count)

# Global singleton for hardware orchestration
system_hardware = HardwareManager()
