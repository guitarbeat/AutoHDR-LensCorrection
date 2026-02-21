import torch
import torch.nn as nn
import torchvision.models as models

class DistortionDetector(nn.Module):
    """
    A ResNet-based CNN to detect lens distortion coefficients.
    
    This model assumes a simplified Brown-Conrady model where we primarily
    aim to predict the radial distortion coefficients (k1, k2, k3) and 
    tangential distortion coefficients (p1, p2) from a single image.
    
    Output dimension is 5: [k1, k2, k3, p1, p2]
    """
    def __init__(self, pretrained=True):
        super(DistortionDetector, self).__init__()
        # Load a pretrained Vision Transformer (ViT) as the feature extractor
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        self.backbone = models.vit_b_16(weights=weights)
        
        # Replace the final classification head to output a flattened spatial
        # displacement map (Flow Field).
        # We need 2 coordinates (dx, dy) for every pixel in a downscaled grid.
        # To keep it computationally feasible for the ViT head, we output a 
        # 14x14 grid of flow vectors, which will be upsampled later.
        # 14 * 14 * 2 = 392
        num_ftrs = self.backbone.heads.head.in_features
        self.backbone.heads.head = nn.Sequential(
            nn.Linear(num_ftrs, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 14 * 14 * 2) # Dense flow field
        )
        
    def forward(self, x):
        """
        Args:
            x: Input image tensor of shape (B, 3, H, W). ViT expects 224x224.
        Returns:
            Tensor of shape (B, 2, 14, 14) containing predicted flow vectors
            for horizontal and vertical displacement.
        """
        flat_flow = self.backbone(x)
        # Reshape into a spatial grid: (Batch, Channels(2), Height(14), Width(14))
        return flat_flow.view(-1, 2, 14, 14)

# Example instantiation/dummy load
def load_model(weights_path=None):
    model = DistortionDetector(pretrained=True)
    if weights_path:
        # model.load_state_dict(torch.load(weights_path))
        pass
    model.eval()
    return model
