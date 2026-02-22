import numpy as np
import cv2

def calculate_mae(img_pred: np.ndarray, img_gt: np.ndarray) -> float:
    """
    Calculates Mean Absolute Error (L1 Loss) between the predicted image
    and the ground truth generated image. This is the primary metric
    for the Kaggle Automatic Lens Correction competition.
    """
    if img_pred.shape != img_gt.shape:
        img_gt = cv2.resize(img_gt, (img_pred.shape[1], img_pred.shape[0]))
    
    # Calculate absolute difference across all channels
    mae = np.mean(np.abs(img_pred.astype(np.float32) - img_gt.astype(np.float32)))
    return float(mae)

def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return 100.0
    pixel_max = 255.0
    return float(20 * np.log10(pixel_max / np.sqrt(mse)))

def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Structural Similarity Index (SSIM).
    Matches the Kaggle secondary scoring system.
    """
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
        
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    var1 = np.var(img1)
    var2 = np.var(img2)
    
    # Flatten across color channels to compute covariance
    cov = np.cov(img1.flatten(), img2.flatten())[0][1]
    
    c1 = (0.01 * 255)**2
    c2 = (0.03 * 255)**2
    
    ssim = ((2 * mu1 * mu2 + c1) * (2 * cov + c2)) / ((mu1**2 + mu2**2 + c1) * (var1 + var2 + c2))
    return float(ssim)

def evaluate_metrics(predicted: np.ndarray, ground_truth: np.ndarray) -> dict:
    """
    Evaluates the prediction against the ground truth using Kaggle metrics.
    Note: Lower MAE is better (0.0 is perfect). Higher SSIM is better (1.0 is perfect).
    """
    mae = calculate_mae(predicted, ground_truth)
    ssim = calculate_ssim(predicted, ground_truth)
    psnr = calculate_psnr(predicted, ground_truth)

    # Convert MAE to a 0-100 Kaggle Score (Approximation where MAE=0 scores 100)
    # If MAE is ~50 (very bad), score approaches 0
    kaggle_scoreEstimate = max(0.0, 100.0 - (mae * 2.0))

    return {
        "mae": round(mae, 4),
        "ssim": round(ssim, 4),
        "psnr": round(psnr, 2),
        "estimated_kaggle_score": round(kaggle_scoreEstimate, 2)
    }
