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
        
        # Replace the final classification head to output distortion coefficients.
        # We need 5 coefficients: k1, k2, k3, p1, p2
        num_ftrs = self.backbone.heads.head.in_features
        self.backbone.heads.head = nn.Sequential(
            nn.Linear(num_ftrs, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 5) # 5 coefficients
        )
        
    def forward(self, x):
        """
        Args:
            x: Input image tensor of shape (B, 3, H, W). ViT expects 224x224.
        Returns:
            Tensor of shape (B, 5) containing predicted distortion coefficients.
        """
        return self.backbone(x)

# Example instantiation/dummy load
def load_model(weights_path=None):
    model = DistortionDetector(pretrained=True)
    if weights_path:
        # model.load_state_dict(torch.load(weights_path))
        pass
    model.eval()
    return model
