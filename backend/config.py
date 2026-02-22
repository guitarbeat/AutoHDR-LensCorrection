"""Centralized runtime configuration for AutoHDR scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import os

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parent.parent

# Load .env from repo root when present.
load_dotenv(dotenv_path=REPO_ROOT / ".env", override=False)
load_dotenv(override=False)


def _resolve_path(raw: Optional[str], default: Path) -> Path:
    if raw is None or raw.strip() == "":
        return default
    return Path(os.path.expanduser(raw)).resolve()


@dataclass(frozen=True)
class AutoHDRConfig:
    repo_root: Path
    data_root: Path
    output_root: Path
    checkpoint_root: Path
    kaggle_mcp_url: str
    kaggle_api_token: Optional[str]
    akash_api_key: Optional[str]

    @property
    def train_dir(self) -> Path:
        return self.data_root / "lens-correction-train-cleaned"

    @property
    def test_dir(self) -> Path:
        return self.data_root / "test-originals"


def get_config() -> AutoHDRConfig:
    data_root = _resolve_path(
        os.getenv("AUTOHDR_DATA_ROOT"),
        Path("/Volumes/Love SSD"),
    )
    output_root = _resolve_path(
        os.getenv("AUTOHDR_OUTPUT_ROOT"),
        data_root / "AutoHDR_Submissions",
    )
    checkpoint_root = _resolve_path(
        os.getenv("AUTOHDR_CHECKPOINT_ROOT"),
        data_root / "AutoHDR_Checkpoints",
    )

    return AutoHDRConfig(
        repo_root=REPO_ROOT,
        data_root=data_root,
        output_root=output_root,
        checkpoint_root=checkpoint_root,
        kaggle_mcp_url=os.getenv("KAGGLE_MCP_URL", "https://www.kaggle.com/mcp"),
        kaggle_api_token=os.getenv("KAGGLE_API_TOKEN"),
        akash_api_key=os.getenv("AKASH_API_KEY"),
    )


def require_existing_dir(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{label} not found: {path}\n"
            "Update your .env or pass an explicit CLI path.\n"
            "Expected env keys: AUTOHDR_DATA_ROOT, AUTOHDR_OUTPUT_ROOT, AUTOHDR_CHECKPOINT_ROOT."
        )
    if not path.is_dir():
        raise NotADirectoryError(f"{label} exists but is not a directory: {path}")
    return path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_kaggle_token(config: AutoHDRConfig) -> str:
    if not config.kaggle_api_token:
        raise RuntimeError(
            "KAGGLE_API_TOKEN is required for Kaggle MCP operations.\n"
            "Set it in your shell or .env."
        )
    return config.kaggle_api_token
