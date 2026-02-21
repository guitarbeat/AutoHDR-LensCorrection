import cv2
import numpy as np
from typing import Tuple

def undistort_image(image: np.ndarray, distortion_coeffs: list[float], camera_matrix=None) -> Tuple[np.ndarray, dict]:
    """
    Applies the inverse distortion geometric correction using OpenCV.
    
    Args:
        image: Original distorted image as a numpy array.
        distortion_coeffs: A list [k1, k2, k3, p1, p2] representing the predicted coefficients.
        camera_matrix: Optional camera intrinsics. If None, it estimates using the image center.
    Returns:
        A tuple of (corrected_image, info_dict).
    """
    h, w = image.shape[:2]
    
    # Estimate a simple intrinsic camera matrix if not provided
    if camera_matrix is None:
        focal_length = w
        center_x = w / 2
        center_y = h / 2
        camera_matrix = np.array([
            [focal_length, 0, center_x],
            [0, focal_length, center_y],
            [0, 0, 1]
        ], dtype=np.float32)
        
    dist_coeffs = np.array(distortion_coeffs, dtype=np.float32)

    # Calculate optimal new camera matrix to keep all pixels
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )

    # Apply undistortion mapping
    corrected_image = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)
    
    # Optionally crop to the Region Of Interest
    x, y, w_roi, h_roi = roi
    if w_roi > 0 and h_roi > 0:
        cropped_image = corrected_image[y:y+h_roi, x:x+w_roi]
    else:
        cropped_image = corrected_image

    info = {
        "original_shape": (w, h),
        "new_shape": (cropped_image.shape[1], cropped_image.shape[0]),
        "roi": roi
    }
    
    return cropped_image, info
