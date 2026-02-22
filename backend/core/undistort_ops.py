"""Shared OpenCV undistortion helpers with cached remap maps."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import cv2
import numpy as np


INTERPOLATION_CHOICES: dict[str, int] = {
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "lanczos4": cv2.INTER_LANCZOS4,
}

BORDER_MODE_CHOICES: dict[str, int] = {
    "constant": cv2.BORDER_CONSTANT,
    "reflect": cv2.BORDER_REFLECT,
    "replicate": cv2.BORDER_REPLICATE,
}


@dataclass(frozen=True)
class MapBundle:
    """Prepared undistortion maps plus metadata for remap."""

    map1: np.ndarray
    map2: np.ndarray
    new_camera_matrix: np.ndarray
    roi: tuple[int, int, int, int]
    interpolation: str
    border_mode: str
    alpha: float
    width: int
    height: int


def build_camera_matrix(width: int, height: int) -> np.ndarray:
    """Estimate camera intrinsics from image dimensions."""
    focal_length = width
    return np.array(
        [
            [focal_length, 0.0, width / 2.0],
            [0.0, focal_length, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def resolve_interpolation(interpolation: str | int) -> tuple[str, int]:
    if isinstance(interpolation, int):
        for name, value in INTERPOLATION_CHOICES.items():
            if value == interpolation:
                return name, value
        return "linear", interpolation
    key = interpolation.strip().lower()
    if key not in INTERPOLATION_CHOICES:
        raise ValueError(f"Unsupported interpolation: {interpolation}")
    return key, INTERPOLATION_CHOICES[key]


def resolve_border_mode(border_mode: str | int) -> tuple[str, int]:
    if isinstance(border_mode, int):
        for name, value in BORDER_MODE_CHOICES.items():
            if value == border_mode:
                return name, value
        return "constant", border_mode
    key = border_mode.strip().lower()
    if key not in BORDER_MODE_CHOICES:
        raise ValueError(f"Unsupported border mode: {border_mode}")
    return key, BORDER_MODE_CHOICES[key]


def _round_key(values: Iterable[float], digits: int = 10) -> tuple[float, ...]:
    return tuple(round(float(v), digits) for v in values)


@lru_cache(maxsize=512)
def _prepare_maps_cached(
    width: int,
    height: int,
    camera_key: tuple[float, ...],
    dist_key: tuple[float, ...],
    alpha: float,
    interpolation: str,
    border_mode: str,
    map_type: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    camera_matrix = np.array(camera_key, dtype=np.float32).reshape(3, 3)
    dist_coeffs = np.array(dist_key, dtype=np.float32)
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        dist_coeffs,
        (width, height),
        alpha,
        (width, height),
    )
    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix,
        dist_coeffs,
        None,
        new_camera_matrix,
        (width, height),
        map_type,
    )
    return map1, map2, new_camera_matrix, tuple(int(v) for v in roi)


def prepare_undistort_maps(
    *,
    width: int,
    height: int,
    dist_coeffs: np.ndarray | list[float] | tuple[float, ...],
    camera_matrix: np.ndarray | None = None,
    alpha: float = 0.0,
    interpolation: str | int = "linear",
    border_mode: str | int = "constant",
    map_type: int = cv2.CV_16SC2,
) -> MapBundle:
    """Prepare and cache undistortion maps for OpenCV remap."""
    interp_name, _ = resolve_interpolation(interpolation)
    border_name, _ = resolve_border_mode(border_mode)

    camera = (
        np.asarray(camera_matrix, dtype=np.float32).reshape(3, 3)
        if camera_matrix is not None
        else build_camera_matrix(width, height)
    )
    dist = np.asarray(dist_coeffs, dtype=np.float32).reshape(-1)

    map1, map2, new_camera_matrix, roi = _prepare_maps_cached(
        width=width,
        height=height,
        camera_key=_round_key(camera.reshape(-1)),
        dist_key=_round_key(dist),
        alpha=float(alpha),
        interpolation=interp_name,
        border_mode=border_name,
        map_type=int(map_type),
    )
    return MapBundle(
        map1=map1,
        map2=map2,
        new_camera_matrix=new_camera_matrix,
        roi=roi,
        interpolation=interp_name,
        border_mode=border_name,
        alpha=float(alpha),
        width=width,
        height=height,
    )


def remap_with_bundle(
    image: np.ndarray,
    bundle: MapBundle,
    *,
    interpolation: str | int | None = None,
    border_mode: str | int | None = None,
) -> np.ndarray:
    """Apply cv2.remap with the cached undistortion bundle."""
    if image.shape[:2] != (bundle.height, bundle.width):
        raise ValueError(
            "Image shape does not match cached map size: "
            f"expected {(bundle.height, bundle.width)}, got {image.shape[:2]}"
        )
    interp_name = bundle.interpolation if interpolation is None else resolve_interpolation(interpolation)[0]
    border_name = bundle.border_mode if border_mode is None else resolve_border_mode(border_mode)[0]
    interp_code = INTERPOLATION_CHOICES[interp_name]
    border_code = BORDER_MODE_CHOICES[border_name]
    return cv2.remap(
        image,
        bundle.map1,
        bundle.map2,
        interpolation=interp_code,
        borderMode=border_code,
    )


def undistort_via_maps(
    image: np.ndarray,
    *,
    dist_coeffs: np.ndarray | list[float] | tuple[float, ...],
    camera_matrix: np.ndarray | None = None,
    alpha: float = 0.0,
    interpolation: str | int = "linear",
    border_mode: str | int = "constant",
    map_type: int = cv2.CV_16SC2,
) -> tuple[np.ndarray, MapBundle]:
    """Undistort an image by preparing/reusing remap maps."""
    height, width = image.shape[:2]
    bundle = prepare_undistort_maps(
        width=width,
        height=height,
        dist_coeffs=dist_coeffs,
        camera_matrix=camera_matrix,
        alpha=alpha,
        interpolation=interpolation,
        border_mode=border_mode,
        map_type=map_type,
    )
    corrected = remap_with_bundle(image, bundle)
    return corrected, bundle


def cache_info() -> str:
    info = _prepare_maps_cached.cache_info()
    return f"hits={info.hits} misses={info.misses} currsize={info.currsize} maxsize={info.maxsize}"
