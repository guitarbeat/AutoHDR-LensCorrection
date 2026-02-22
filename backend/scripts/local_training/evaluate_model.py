#!/usr/bin/env python3
"""Evaluate a trained model checkpoint on validation data."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from backend.config import ensure_dir, get_config, require_existing_dir
from backend.core.dataloader import get_dataloaders
from backend.evaluation.metrics import calculate_mae, calculate_psnr, calculate_ssim
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
    parser = argparse.ArgumentParser(description="Evaluate model on validation split")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--data-root", default=str(cfg.data_root))
    parser.add_argument("--output-root", default=str(cfg.output_root))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--img-size", type=int, default=None, help="Override checkpoint image size"
    )
    parser.add_argument(
        "--max-val", type=int, default=None, help="Limit validation samples"
    )
    parser.add_argument(
        "--device", choices=["auto", "cuda", "mps", "cpu"], default="auto"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    output_root = ensure_dir(Path(args.output_root).expanduser().resolve())
    eval_dir = ensure_dir(output_root / "evaluation")
    require_existing_dir(data_root, "Dataset root")
    require_existing_dir(
        data_root / "lens-correction-train-cleaned", "Training images directory"
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = str(checkpoint.get("model_name", "micro_unet")).lower()
    model_class = MODEL_REGISTRY.get(model_name, MODEL_REGISTRY["micro_unet"])
    model = model_class(in_channels=3, out_channels=3)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    img_size = int(args.img_size or checkpoint.get("img_size", 256))
    normalize = bool(checkpoint.get("normalize", False))

    _, val_loader, _ = get_dataloaders(
        root_dir=str(data_root),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=img_size,
        normalize=normalize,
        max_train=1,
        max_val=args.max_val,
    )

    maes: list[float] = []
    psnrs: list[float] = []
    ssims: list[float] = []
    t0 = time.time()
    with torch.no_grad():
        for batch in val_loader:
            original = batch["original"].to(device)
            generated = batch["generated"].to(device)
            predicted = model(original)

            pred_np = predicted.cpu().numpy()
            gt_np = generated.cpu().numpy()
            for i in range(pred_np.shape[0]):
                pred_img = np.clip(
                    pred_np[i].transpose(1, 2, 0) * 255.0, 0, 255
                ).astype(np.uint8)
                gt_img = np.clip(gt_np[i].transpose(1, 2, 0) * 255.0, 0, 255).astype(
                    np.uint8
                )
                maes.append(calculate_mae(pred_img, gt_img))
                psnrs.append(calculate_psnr(pred_img, gt_img))
                ssims.append(calculate_ssim(pred_img, gt_img))

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint_path),
        "model_name": model_name,
        "img_size": img_size,
        "normalize": normalize,
        "device": str(device),
        "num_samples": len(maes),
        "metrics": {
            "mae_mean": round(float(np.mean(maes)) if maes else 0.0, 6),
            "mae_std": round(float(np.std(maes)) if maes else 0.0, 6),
            "psnr_mean": round(float(np.mean(psnrs)) if psnrs else 0.0, 6),
            "psnr_std": round(float(np.std(psnrs)) if psnrs else 0.0, 6),
            "ssim_mean": round(float(np.mean(ssims)) if ssims else 0.0, 6),
            "ssim_std": round(float(np.std(ssims)) if ssims else 0.0, 6),
        },
        "elapsed_s": round(time.time() - t0, 3),
    }

    out_name = f"{checkpoint_path.stem}_metrics.json"
    out_path = eval_dir / out_name
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Saved evaluation summary: {out_path}")


if __name__ == "__main__":
    main()
