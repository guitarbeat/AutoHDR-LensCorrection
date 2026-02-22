#!/usr/bin/env python3
"""Backward-compatible wrapper for build_real_fusion_variants.py."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    print(
        "[deprecation] build_real_oracle_variants.py is deprecated; "
        "use build_real_fusion_variants.py.",
        file=sys.stderr,
    )
    runpy.run_path(
        str(Path(__file__).resolve().with_name("build_real_fusion_variants.py")),
        run_name="__main__",
    )
