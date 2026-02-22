#!/usr/bin/env python3
"""Gradio demo for heuristic lens distortion correction."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import gradio as gr
import numpy as np

from backend.config import get_config
from backend.core.distortion import undistort_image


def load_coefficients() -> tuple[float, float, Optional[Path]]:
    cfg = get_config()
    candidates = [
        cfg.output_root / "best_coefficients.txt",
        cfg.repo_root / "backend" / "outputs" / "best_coefficients.txt",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            lines = path.read_text().splitlines()
            values = {}
            for line in lines:
                if "=" in line:
                    k, v = line.split("=", 1)
                    values[k.strip()] = float(v.strip())
            if "k1" in values and "k2" in values:
                return values["k1"], values["k2"], path
        except Exception:
            continue
    return -0.17, 0.35, None


def build_examples() -> list[list[str]]:
    cfg = get_config()
    test_dir = cfg.test_dir
    if not test_dir.exists():
        return []
    jpgs = sorted(test_dir.glob("*.jpg"))
    return [[str(p)] for p in jpgs[:4]]


K1, K2, COEFF_SOURCE = load_coefficients()
if COEFF_SOURCE:
    print(f"Using coefficients from {COEFF_SOURCE}: k1={K1:.4f}, k2={K2:.4f}")
else:
    print(f"Using fallback coefficients: k1={K1:.4f}, k2={K2:.4f}")


def _draw_labeled_pair(
    left_rgb: np.ndarray, right_rgb: np.ndarray, left_label: str, right_label: str
) -> np.ndarray:
    h, w = left_rgb.shape[:2]
    if right_rgb.shape[:2] != (h, w):
        right_rgb = cv2.resize(right_rgb, (w, h), interpolation=cv2.INTER_LINEAR)

    pair = np.hstack([left_rgb, right_rgb])
    label_bar_h = 44
    canvas = np.zeros((h + label_bar_h, pair.shape[1], 3), dtype=np.uint8)
    canvas[:h] = pair
    canvas[h:] = 18

    cv2.putText(
        canvas,
        left_label,
        (14, h + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        right_label,
        (w + 14, h + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )
    return canvas


def _build_diff_heatmap(
    original_rgb: np.ndarray, corrected_rgb: np.ndarray
) -> tuple[np.ndarray, float, float]:
    h, w = original_rgb.shape[:2]
    if corrected_rgb.shape[:2] != (h, w):
        corrected_rgb = cv2.resize(
            corrected_rgb, (w, h), interpolation=cv2.INTER_LINEAR
        )

    diff_rgb = cv2.absdiff(original_rgb, corrected_rgb)
    diff_gray = cv2.cvtColor(diff_rgb, cv2.COLOR_RGB2GRAY)
    if int(diff_gray.max()) > 0:
        diff_norm = cv2.normalize(diff_gray, None, 0, 255, cv2.NORM_MINMAX)
    else:
        diff_norm = diff_gray

    heatmap_bgr = cv2.applyColorMap(diff_norm, cv2.COLORMAP_TURBO)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    return heatmap_rgb, float(diff_gray.mean()), float(diff_gray.max())


def _build_radial_curve_plot(
    k1: float, k2: float, width: int = 640, height: int = 360
) -> np.ndarray:
    canvas = np.full((height, width, 3), 252, dtype=np.uint8)
    margin = 56
    graph_w = width - 2 * margin
    graph_h = height - 2 * margin
    y_max = 1.35

    cv2.rectangle(
        canvas, (margin, margin), (width - margin, height - margin), (220, 220, 220), 1
    )

    radii = np.linspace(0.0, 1.0, 220)
    distorted = radii * (1.0 + k1 * radii**2 + k2 * radii**4)
    distorted = np.clip(distorted, 0.0, y_max)

    x = margin + (radii * graph_w).astype(np.int32)
    y_identity = (height - margin - (radii / y_max) * graph_h).astype(np.int32)
    y_distorted = (height - margin - (distorted / y_max) * graph_h).astype(np.int32)

    identity_pts = np.column_stack((x, y_identity)).reshape((-1, 1, 2))
    curve_pts = np.column_stack((x, y_distorted)).reshape((-1, 1, 2))
    cv2.polylines(
        canvas, [identity_pts], isClosed=False, color=(170, 170, 170), thickness=2
    )
    cv2.polylines(canvas, [curve_pts], isClosed=False, color=(36, 87, 255), thickness=3)

    cv2.putText(
        canvas,
        "Identity",
        (margin + 10, margin + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (120, 120, 120),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Distorted curve",
        (margin + 110, margin + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (36, 87, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Input radius (normalized)",
        (margin + 120, height - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (70, 70, 70),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Output radius",
        (8, margin + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (70, 70, 70),
        1,
        cv2.LINE_AA,
    )
    return canvas


def reset_coefficients() -> tuple[float, float]:
    return K1, K2


def process_image(input_image: np.ndarray, k1: float, k2: float):
    radial_plot = _build_radial_curve_plot(k1, k2)
    if input_image is None:
        return None, None, None, radial_plot, "Please upload an image."

    img_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    coeffs = [k1, k2, 0.0, 0.0, 0.0]
    corrected_bgr, info = undistort_image(img_bgr, coeffs)
    corrected_rgb = cv2.cvtColor(corrected_bgr, cv2.COLOR_BGR2RGB)
    resized_corrected = cv2.resize(
        corrected_rgb,
        (input_image.shape[1], input_image.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    comparison = _draw_labeled_pair(
        input_image, resized_corrected, "Original", "Corrected"
    )
    heatmap, mean_abs_error, max_abs_error = _build_diff_heatmap(
        input_image, resized_corrected
    )

    edge_before = cv2.Canny(cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY), 100, 200)
    edge_after = cv2.Canny(
        cv2.cvtColor(resized_corrected, cv2.COLOR_RGB2GRAY), 100, 200
    )
    edge_before_density = float(np.mean(edge_before > 0))
    edge_after_density = float(np.mean(edge_after > 0))
    kept_area_ratio = (info["new_shape"][0] * info["new_shape"][1]) / (
        info["original_shape"][0] * info["original_shape"][1]
    )

    coeff_source = f"`{COEFF_SOURCE}`" if COEFF_SOURCE else "fallback defaults"
    stats = f"""
### Correction Applied
- **Algorithm**: Brown-Conrady OpenCV heuristic
- **Default coefficient source**: {coeff_source}
- **Active k1**: {k1:.4f}
- **Active k2**: {k2:.4f}

### Image Info
- **Original Size**: {info['original_shape'][0]}x{info['original_shape'][1]}
- **Corrected Size**: {info['new_shape'][0]}x{info['new_shape'][1]}
- **Retained area after crop**: {kept_area_ratio * 100:.2f}%

### Diagnostics
- **Mean absolute pixel error** (resized comparison): {mean_abs_error:.2f}
- **Max absolute pixel error**: {max_abs_error:.2f}
- **Edge density before**: {edge_before_density:.4f}
- **Edge density after**: {edge_after_density:.4f}
"""
    return corrected_rgb, comparison, heatmap, radial_plot, stats


def create_demo() -> gr.Blocks:
    examples = build_examples()
    with gr.Blocks(theme=gr.themes.Soft()) as app:
        gr.Markdown(
            """
            # AutoHDR: Automated Lens Distortion Correction
            Upload a wide-angle real-estate image to apply geometric undistortion, then inspect
            before/after differences and the radial curve induced by `k1`/`k2`.
            """
        )

        with gr.Row():
            with gr.Column(scale=4):
                input_img = gr.Image(label="Distorted Input", type="numpy")
                with gr.Row():
                    k1_slider = gr.Slider(
                        minimum=-1.0,
                        maximum=1.0,
                        value=K1,
                        step=0.005,
                        label="k1",
                        info="Primary radial term",
                    )
                    k2_slider = gr.Slider(
                        minimum=-1.0,
                        maximum=1.0,
                        value=K2,
                        step=0.005,
                        label="k2",
                        info="Secondary radial term",
                    )
                with gr.Row():
                    submit_btn = gr.Button(
                        "Apply Geometric Correction", variant="primary"
                    )
                    reset_btn = gr.Button("Reset to Best Coefficients")
                if examples:
                    gr.Markdown("### Example images")
                    gr.Examples(examples=examples, inputs=input_img)
                else:
                    gr.Markdown(
                        "### Example images unavailable\n"
                        "Set `AUTOHDR_DATA_ROOT` so `test-originals/` is discoverable."
                    )
            with gr.Column(scale=5):
                output_img = gr.Image(label="Corrected Output")
                output_stats = gr.Markdown(label="Correction Details")

        with gr.Row():
            comparison_img = gr.Image(label="Before vs Corrected")
            heatmap_img = gr.Image(label="Absolute Difference Heatmap")

        radial_curve_img = gr.Image(label="Radial Distortion Curve")

        submit_btn.click(
            fn=process_image,
            inputs=[input_img, k1_slider, k2_slider],
            outputs=[
                output_img,
                comparison_img,
                heatmap_img,
                radial_curve_img,
                output_stats,
            ],
        )
        reset_btn.click(fn=reset_coefficients, outputs=[k1_slider, k2_slider])
    return app


if __name__ == "__main__":
    create_demo().launch(server_name="0.0.0.0", server_port=7860, share=False)
