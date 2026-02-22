#!/usr/bin/env python3
"""Train U-Net variants for lens distortion correction."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Type

import torch
import torch.nn as nn
import torch.optim as optim

from backend.config import ensure_dir, get_config, require_existing_dir
from backend.core.dataloader import get_dataloaders


class ConvBlock(nn.Module):
    """Two convolutions + BatchNorm + ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetSmall(nn.Module):
    """Larger U-Net variant."""

    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        self.bottleneck = ConvBlock(512, 1024)
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(1024, 512, 3, padding=1),
        )
        self.dec4 = ConvBlock(1024, 512)
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(512, 256, 3, padding=1),
        )
        self.dec3 = ConvBlock(512, 256)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(256, 128, 3, padding=1),
        )
        self.dec2 = ConvBlock(256, 128)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(128, 64, 3, padding=1),
        )
        self.dec1 = ConvBlock(128, 64)
        self.out_conv = nn.Conv2d(64, out_channels, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.out_conv(d1))


class MicroUNet(nn.Module):
    """Lightweight U-Net variant for faster iteration."""

    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, 16)
        self.enc2 = ConvBlock(16, 32)
        self.enc3 = ConvBlock(32, 64)
        self.enc4 = ConvBlock(64, 128)
        self.bottleneck = ConvBlock(128, 256)
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(256, 128, 3, padding=1),
        )
        self.dec4 = ConvBlock(256, 128)
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(128, 64, 3, padding=1),
        )
        self.dec3 = ConvBlock(128, 64)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, 32, 3, padding=1),
        )
        self.dec2 = ConvBlock(64, 32)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(32, 16, 3, padding=1),
        )
        self.dec1 = ConvBlock(32, 16)
        self.out_conv = nn.Conv2d(16, out_channels, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.out_conv(d1))


MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "micro_unet": MicroUNet,
    "unet_small": UNetSmall,
}


class SobelFilter(nn.Module):
    def __init__(self):
        super().__init__()
        gx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 4.0
        gy = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]) / 4.0
        self.gx = gx.view(1, 1, 3, 3)
        self.gy = gy.view(1, 1, 3, 3)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        gx = self.gx.to(img.device)
        gy = self.gy.to(img.device)
        gray = (
            0.2989 * img[:, 0:1, :, :]
            + 0.5870 * img[:, 1:2, :, :]
            + 0.1140 * img[:, 2:3, :, :]
        )
        grad_x = nn.functional.conv2d(gray, gx, padding=1)
        grad_y = nn.functional.conv2d(gray, gy, padding=1)
        return torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)


class CombinedLoss(nn.Module):
    """L1 loss + Sobel edge loss."""

    def __init__(self, l1_weight: float = 1.0, edge_weight: float = 0.5):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.sobel = SobelFilter()
        self.l1_weight = l1_weight
        self.edge_weight = edge_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_l1 = self.l1(pred, target)
        pred_edges = self.sobel(pred)
        target_edges = self.sobel(target)
        loss_edge = self.l1(pred_edges, target_edges)
        return self.l1_weight * loss_l1 + self.edge_weight * loss_edge


def get_device(allow_mps: bool = False) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if (
        allow_mps
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        original = batch["original"].to(device)
        generated = batch["generated"].to(device)
        optimizer.zero_grad()
        predicted = model(original)
        loss = criterion(predicted, generated)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    num_batches = 0

    for batch in loader:
        original = batch["original"].to(device)
        generated = batch["generated"].to(device)
        predicted = model(original)
        loss = criterion(predicted, generated)
        mae_255 = (predicted - generated).abs().mean().item() * 255.0
        total_loss += float(loss.item())
        total_mae += mae_255
        num_batches += 1

    return total_loss / max(num_batches, 1), total_mae / max(num_batches, 1)


def checkpoint_payload(
    *,
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    val_loss: float,
    val_mae_255: float,
    model_name: str,
    img_size: int,
    normalize: bool,
) -> dict:
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "val_mae_255": val_mae_255,
        "model_name": model_name,
        "img_size": img_size,
        "normalize": normalize,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def parse_args() -> argparse.Namespace:
    cfg = get_config()
    parser = argparse.ArgumentParser(
        description="Train U-Net for lens distortion correction"
    )
    parser.add_argument(
        "--data-root",
        default=str(cfg.data_root),
        help="Dataset root containing lens-correction-train-cleaned/ and test-originals/",
    )
    parser.add_argument(
        "--output-dir",
        default=str(cfg.checkpoint_root),
        help="Directory for checkpoints and training history",
    )
    parser.add_argument(
        "--model", choices=sorted(MODEL_REGISTRY.keys()), default="micro_unet"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--normalize", action="store_true", help="Apply ImageNet normalization"
    )
    parser.add_argument(
        "--max-train", type=int, default=None, help="Limit training samples"
    )
    parser.add_argument(
        "--max-val", type=int, default=None, help="Limit validation samples"
    )
    parser.add_argument(
        "--overfit", action="store_true", help="Use same tiny split for train/val"
    )
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument(
        "--allow-mps", action="store_true", help="Allow MPS device usage"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = ensure_dir(Path(args.output_dir).expanduser().resolve())
    require_existing_dir(data_root, "Dataset root")
    require_existing_dir(
        data_root / "lens-correction-train-cleaned", "Training images directory"
    )
    require_existing_dir(data_root / "test-originals", "Test images directory")

    device = get_device(allow_mps=args.allow_mps)
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")

    if args.overfit:
        max_train = args.max_train or 10
        max_val = max_train
        print(f"Overfit mode: train={max_train}, val={max_val}")
    else:
        max_train = args.max_train
        max_val = args.max_val

    train_loader, val_loader, _ = get_dataloaders(
        root_dir=str(data_root),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        normalize=args.normalize,
        max_train=max_train,
        max_val=max_val,
    )
    print(
        f"Train: {len(train_loader.dataset)} samples, Val: {len(val_loader.dataset)} samples"
    )

    model_class = MODEL_REGISTRY[args.model]
    model = model_class(in_channels=3, out_channels=3).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model} ({param_count:,} parameters)")

    criterion = CombinedLoss(l1_weight=1.0)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    history: list[dict] = []
    print(f"Starting training for {args.epochs} epochs")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_mae = validate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        log_entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_mae_255": round(val_mae, 4),
            "lr": round(lr, 10),
            "time_s": round(elapsed, 2),
        }
        history.append(log_entry)

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
            f"Val MAE(255): {val_mae:.4f} | LR: {lr:.7f} | {elapsed:.1f}s"
        )

        payload = checkpoint_payload(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            val_loss=val_loss,
            val_mae_255=val_mae,
            model_name=args.model,
            img_size=args.img_size,
            normalize=args.normalize,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = output_dir / "best_model.pt"
            torch.save(payload, best_path)
            print(f"  New best model saved: {best_path} (val_loss={val_loss:.6f})")

        if epoch % max(args.save_every, 1) == 0:
            ckpt_path = output_dir / f"checkpoint_epoch{epoch}.pt"
            torch.save(payload, ckpt_path)

        if args.overfit and train_loss < 0.001:
            print(
                f"Overfit threshold reached (train_loss={train_loss:.6f}); stopping early"
            )
            break

    history_path = output_dir / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2))
    print("=" * 70)
    print("Training complete")
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"History: {history_path}")
    print(f"Best checkpoint: {output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
