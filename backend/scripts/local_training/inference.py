#!/usr/bin/env python3
"""Run model inference on test images and create a submission zip."""

from __future__ import annotations

import argparse
import json
import time
import zipfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from backend.config import ensure_dir, get_config, require_existing_dir
from backend.scripts.local_training.train import MODEL_REGISTRY


def choose_device(prefer: str) -> torch.device:
    if prefer == "cuda":
        return torch.device("cuda")
    if prefer == "mps":
        return torch.device("mps")
    if prefer == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    cfg = get_config()
    parser = argparse.ArgumentParser(description="U-Net inference for lens correction")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--test-dir", default=str(cfg.test_dir))
    parser.add_argument(
        "--output-root",
        default=str(cfg.output_root),
        help="Root output directory for deterministic inference artifacts",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional deterministic run name (default: checkpoint stem)",
    )
    parser.add_argument("--img-size", type=int, default=None, help="Override checkpoint image size")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test images")
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Inference device selection",
    )
    return parser.parse_args()


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> dict:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location=device)


def load_model(checkpoint: dict, device: torch.device) -> tuple[torch.nn.Module, str]:
    model_name = str(checkpoint.get("model_name", "micro_unet")).lower()
    model_class = MODEL_REGISTRY.get(model_name, MODEL_REGISTRY["micro_unet"])
    model = model_class(in_channels=3, out_channels=3)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, model_name


def correct_image(
    model: torch.nn.Module,
    img_path: Path,
    device: torch.device,
    img_size: int,
) -> np.ndarray:
    original = cv2.imread(str(img_path))
    if original is None:
        raise ValueError(f"Could not read image: {img_path}")

    orig_h, orig_w = original.shape[:2]
    pil_img = Image.fromarray(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    tfm = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
    tensor = tfm(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        predicted = model(tensor)

    pred_np = predicted[0].cpu().numpy().transpose(1, 2, 0)
    pred_np = np.clip(pred_np * 255.0, 0, 255).astype(np.uint8)
    pred_bgr = cv2.cvtColor(pred_np, cv2.COLOR_RGB2BGR)

    if pred_bgr.shape[0] != orig_h or pred_bgr.shape[1] != orig_w:
        pred_bgr = cv2.resize(pred_bgr, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)

    return pred_bgr


def build_output_layout(output_root: Path, checkpoint_path: Path, run_name: Optional[str]) -> dict:
    name = run_name or checkpoint_path.stem
    run_dir = ensure_dir(output_root / "inference" / name)
    corrected_dir = ensure_dir(run_dir / "corrected")
    zip_path = run_dir / f"{name}.zip"
    summary_path = run_dir / "summary.json"
    return {
        "name": name,
        "run_dir": run_dir,
        "corrected_dir": corrected_dir,
        "zip_path": zip_path,
        "summary_path": summary_path,
    }


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    test_dir = Path(args.test_dir).expanduser().resolve()
    output_root = ensure_dir(Path(args.output_root).expanduser().resolve())

    checkpoint = load_checkpoint(checkpoint_path, device)
    model, model_name = load_model(checkpoint, device)
    img_size = int(args.img_size or checkpoint.get("img_size", 256))
    layout = build_output_layout(output_root, checkpoint_path, args.run_name)

    require_existing_dir(test_dir, "Test image directory")
    test_files = sorted([p for p in test_dir.iterdir() if p.suffix.lower() == ".jpg"])
    if args.limit is not None:
        test_files = test_files[: args.limit]

    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Model: {model_name}")
    print(f"Image size: {img_size}")
    print(f"Test images: {len(test_files)}")
    print(f"Run directory: {layout['run_dir']}")

    t0 = time.time()
    for idx, img_path in enumerate(test_files, start=1):
        corrected = correct_image(model, img_path, device, img_size)
        out_path = layout["corrected_dir"] / img_path.name
        cv2.imwrite(str(out_path), corrected, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if idx % 100 == 0 or idx == len(test_files):
            elapsed = time.time() - t0
            rate = elapsed / max(idx, 1)
            remaining = rate * (len(test_files) - idx)
            print(f"  [{idx}/{len(test_files)}] ETA: {remaining:.0f}s")

    with zipfile.ZipFile(layout["zip_path"], "w", zipfile.ZIP_STORED) as zf:
        for img_path in test_files:
            corrected_path = layout["corrected_dir"] / img_path.name
            if corrected_path.exists():
                zf.write(corrected_path, img_path.name)

    summary = {
        "checkpoint": str(checkpoint_path),
        "model_name": model_name,
        "img_size": img_size,
        "device": str(device),
        "test_dir": str(test_dir),
        "num_images": len(test_files),
        "run_dir": str(layout["run_dir"]),
        "corrected_dir": str(layout["corrected_dir"]),
        "zip_path": str(layout["zip_path"]),
        "elapsed_s": round(time.time() - t0, 3),
    }
    layout["summary_path"].write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
