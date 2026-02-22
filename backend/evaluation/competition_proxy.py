"""Competition-aligned proxy metrics for geometric correction evaluation."""

from __future__ import annotations

import math

import cv2
import numpy as np


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _canny_union(gray: np.ndarray) -> np.ndarray:
    edges_1 = cv2.Canny(gray, 50, 150)
    edges_2 = cv2.Canny(gray, 80, 160)
    edges_3 = cv2.Canny(gray, 110, 220)
    return cv2.bitwise_or(cv2.bitwise_or(edges_1, edges_2), edges_3)


def edge_similarity_multiscale(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_gray = _to_gray(pred)
    gt_gray = _to_gray(gt)
    pred_edges = _canny_union(pred_gray) > 0
    gt_edges = _canny_union(gt_gray) > 0
    tp = int(np.logical_and(pred_edges, gt_edges).sum())
    fp = int(np.logical_and(pred_edges, ~gt_edges).sum())
    fn = int(np.logical_and(~pred_edges, gt_edges).sum())
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0
    if tp == 0:
        return 0.0
    return float((2.0 * tp) / (2.0 * tp + fp + fn))


def _line_hist(gray: np.ndarray, bins: int = 18) -> np.ndarray:
    # Prefer LSD for robust real-estate line extraction. Fallback to HoughP.
    angles: list[float] = []
    if hasattr(cv2, "createLineSegmentDetector"):
        lsd = cv2.createLineSegmentDetector()
        lines = lsd.detect(gray)[0]
        if lines is not None:
            for item in lines:
                x1, y1, x2, y2 = item[0]
                angle = math.degrees(math.atan2(float(y2 - y1), float(x2 - x1)))
                angles.append((angle + 180.0) % 180.0)
    if not angles:
        edges = _canny_union(gray)
        min_line = max(8, min(gray.shape[0], gray.shape[1]) // 6)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180.0,
            threshold=30,
            minLineLength=min_line,
            maxLineGap=6,
        )
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = math.degrees(math.atan2(float(y2 - y1), float(x2 - x1)))
                angles.append((angle + 180.0) % 180.0)
    if not angles:
        return np.zeros(bins, dtype=np.float32)
    hist, _ = np.histogram(np.array(angles), bins=bins, range=(0.0, 180.0))
    hist = hist.astype(np.float32)
    total = float(hist.sum())
    if total <= 0.0:
        return np.zeros(bins, dtype=np.float32)
    return hist / total


def line_orientation_loss(pred: np.ndarray, gt: np.ndarray, bins: int = 18) -> float:
    pred_hist = _line_hist(_to_gray(pred), bins=bins)
    gt_hist = _line_hist(_to_gray(gt), bins=bins)
    pred_sum = float(pred_hist.sum())
    gt_sum = float(gt_hist.sum())
    if pred_sum == 0.0 and gt_sum == 0.0:
        return 0.0
    if pred_sum == 0.0 or gt_sum == 0.0:
        return 1.0
    return float(0.5 * np.abs(pred_hist - gt_hist).sum())


def _grad_hist(gray: np.ndarray, bins: int = 36) -> np.ndarray:
    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    mag = mag.astype(np.float32)
    threshold = float(np.percentile(mag, 60.0)) if mag.size else 0.0
    mask = mag > max(threshold, 1e-6)
    if not np.any(mask):
        return np.zeros(bins, dtype=np.float32)
    hist, _ = np.histogram(
        ang[mask],
        bins=bins,
        range=(0.0, 360.0),
        weights=mag[mask],
    )
    hist = hist.astype(np.float32)
    total = float(hist.sum())
    if total <= 0.0:
        return np.zeros(bins, dtype=np.float32)
    return hist / total


def gradient_orientation_loss(pred: np.ndarray, gt: np.ndarray, bins: int = 36) -> float:
    pred_hist = _grad_hist(_to_gray(pred), bins=bins)
    gt_hist = _grad_hist(_to_gray(gt), bins=bins)
    pred_norm = float(np.linalg.norm(pred_hist))
    gt_norm = float(np.linalg.norm(gt_hist))
    if pred_norm == 0.0 and gt_norm == 0.0:
        return 0.0
    if pred_norm == 0.0 or gt_norm == 0.0:
        return 1.0
    cosine = float(np.dot(pred_hist, gt_hist) / (pred_norm * gt_norm + 1e-8))
    return float(np.clip(1.0 - cosine, 0.0, 1.0))


def ssim_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_gray = _to_gray(pred)
    gt_gray = _to_gray(gt)
    if pred_gray.shape != gt_gray.shape:
        gt_gray = cv2.resize(gt_gray, (pred_gray.shape[1], pred_gray.shape[0]))
    try:
        from skimage.metrics import structural_similarity

        value = float(structural_similarity(pred_gray, gt_gray, data_range=255))
        return float(np.clip(value, 0.0, 1.0))
    except Exception:
        x = pred_gray.astype(np.float32)
        y = gt_gray.astype(np.float32)
        mu_x = float(x.mean())
        mu_y = float(y.mean())
        sigma_x = float(((x - mu_x) ** 2).mean())
        sigma_y = float(((y - mu_y) ** 2).mean())
        sigma_xy = float(((x - mu_x) * (y - mu_y)).mean())
        c1 = (0.01 * 255.0) ** 2
        c2 = (0.03 * 255.0) ** 2
        denom = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
        if denom <= 0.0:
            return 0.0
        return float(np.clip(((2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)) / denom, 0.0, 1.0))


def normalized_mae(pred: np.ndarray, gt: np.ndarray) -> float:
    if pred.shape != gt.shape:
        gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]))
    return float(np.mean(np.abs(pred.astype(np.float32) - gt.astype(np.float32))) / 255.0)
