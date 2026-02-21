import numpy as np
import cv2

def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    # Resize img2 to match img1 for calculation purposes if needed
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    pixel_max = 255.0
    return 20 * np.log10(pixel_max / np.sqrt(mse))

def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    # A simplified SSIM placeholder using basic variance metrics
    # In practice, usually from skimage.metrics import structural_similarity
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    var1 = np.var(img1)
    var2 = np.var(img2)
    cov = np.cov(img1.flatten(), img2.flatten())[0][1]
    
    c1 = (0.01 * 255)**2
    c2 = (0.03 * 255)**2
    
    ssim = ((2 * mu1 * mu2 + c1) * (2 * cov + c2)) / ((mu1**2 + mu2**2 + c1) * (var1 + var2 + c2))
    return float(ssim)

def evaluate_metrics(original: np.ndarray, corrected: np.ndarray) -> dict:
    """
    Evaluates the geometric accuracy of the corrected image against the original
    (distorted) image or an ideal reference if available.
    """
    
    # Calculate base PSNR and SSIM
    # In a real scenario, this would compare corrected to a known ground-truth.
    psnr = calculate_psnr(original, corrected)
    ssim = calculate_ssim(original, corrected)

    # 1. Edge alignment
    # Approximate using edge detection maps correlation
    edges_orig = cv2.Canny(original, 100, 200)
    edges_corr = cv2.Canny(corrected, 100, 200)
    if edges_orig.shape != edges_corr.shape:
        edges_corr = cv2.resize(edges_corr, (edges_orig.shape[1], edges_orig.shape[0]))
    edge_alignment = np.corrcoef(edges_orig.flatten(), edges_corr.flatten())[0, 1]

    # 2. Line straightness
    # Approximate: HoughLines to find strong lines in the corrected image
    lines = cv2.HoughLines(edges_corr, 1, np.pi / 180, 200)
    line_straightness_score = float(len(lines) if lines is not None else 0) / 100.0
    
    # 3. Gradient orientation
    # Compare Sobel gradients
    sobel_orig_x = cv2.Sobel(original, cv2.CV_64F, 1, 0, ksize=3)
    sobel_corr_x = cv2.Sobel(corrected, cv2.CV_64F, 1, 0, ksize=3)
    if sobel_orig_x.shape != sobel_corr_x.shape:
        sobel_corr_x = cv2.resize(sobel_corr_x, (sobel_orig_x.shape[1], sobel_orig_x.shape[0]))
    gradient_orientation = float(np.corrcoef(np.abs(sobel_orig_x).flatten(), np.abs(sobel_corr_x).flatten())[0, 1])

    # 4. Structural similarity
    structural_similarity = ssim
    
    # 5. Pixel accuracy
    # Normalized MSE or PSNR relative to 1.0 (inverted)
    pixel_accuracy = min(max(psnr / 50.0, 0.0), 1.0) # Scale PSNR 0-50 to 0.0-1.0

    return {
        "psnr": float(psnr),
        "rpe": float(1.0 - structural_similarity), # Surrogate for reprojection error
        "edge_alignment": float(max(0, edge_alignment)),
        "line_straightness": min(line_straightness_score, 1.0),
        "gradient_orientation": float(max(0, gradient_orientation)),
        "structural_similarity": float(max(0, structural_similarity)),
        "pixel_accuracy": float(pixel_accuracy)
    }
