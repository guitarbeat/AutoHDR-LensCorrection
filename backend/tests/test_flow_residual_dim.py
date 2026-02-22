import numpy as np
import zipfile
from pathlib import Path

from backend.scripts.heuristics.heuristic_flow_residual_dim import (
    BORDER_RATIO_MAX,
    aggregate_residual_flow,
    build_zip_from_render_dirs,
    compute_black_border_ratio,
    resize_flow_to_image,
    should_accept_dimension,
)


def test_resize_flow_to_image_scales_components():
    flow = np.zeros((2, 2, 2), dtype=np.float32)
    flow[..., 0] = 1.0
    flow[..., 1] = 0.5

    resized = resize_flow_to_image(flow, height=4, width=4)

    assert resized.shape == (4, 4, 2)
    assert np.allclose(resized[..., 0], 2.0, atol=1e-5)
    assert np.allclose(resized[..., 1], 1.0, atol=1e-5)


def test_rejected_dimension_falls_back_to_base_reason():
    accepted, reason = should_accept_dimension(
        proxy_gain=0.0,
        border_ratio=0.0,
        warp_risk=0.0,
        flow_var=0.0,
        min_proxy_gain=0.002,
        max_border_ratio=0.002,
        max_warp_risk=0.35,
        max_flow_var=3.5,
    )

    assert accepted is False
    assert reason == "reject_low_proxy_gain"


def test_aggregate_residual_flow_is_deterministic_for_fixed_inputs():
    rng = np.random.default_rng(42)
    flow_stack = [rng.normal(loc=0.0, scale=0.2, size=(8, 12, 2)).astype(np.float32) for _ in range(5)]

    out_a = aggregate_residual_flow(flow_stack, flow_max_mag=40.0)
    out_b = aggregate_residual_flow(flow_stack, flow_max_mag=40.0)

    for idx in range(4):
        if isinstance(out_a[idx], np.ndarray):
            assert np.allclose(out_a[idx], out_b[idx], atol=1e-7)
        else:
            assert out_a[idx] == out_b[idx]


def test_border_ratio_guardrail_triggers_rejection():
    img = np.full((20, 20, 3), 255, dtype=np.uint8)
    img[:10, :20, :] = 0
    border_ratio = compute_black_border_ratio(img)

    assert border_ratio > BORDER_RATIO_MAX

    accepted, reason = should_accept_dimension(
        proxy_gain=0.01,
        border_ratio=border_ratio,
        warp_risk=0.0,
        flow_var=0.0,
        min_proxy_gain=0.002,
        max_border_ratio=BORDER_RATIO_MAX,
        max_warp_risk=0.35,
        max_flow_var=3.5,
    )

    assert accepted is False
    assert reason == "reject_border_ratio"


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def test_best_effort_repairs_missing_rendered_image_with_original(tmp_path: Path):
    expected_names = ["img_a.jpg", "img_b.jpg"]
    primary_dir = tmp_path / "candidate"
    base_dir = tmp_path / "baseline"
    originals_dir = tmp_path / "test_originals"
    out_zip = tmp_path / "out.zip"

    _write_bytes(originals_dir / "img_a.jpg", b"orig-a")
    _write_bytes(originals_dir / "img_b.jpg", b"orig-b")

    # img_a is intentionally missing from rendered outputs (simulated unreadable/failed render).
    _write_bytes(primary_dir / "img_b.jpg", b"cand-b")
    _write_bytes(base_dir / "img_b.jpg", b"base-b")

    stats = build_zip_from_render_dirs(
        output_zip=out_zip,
        expected_names=expected_names,
        primary_dir=primary_dir,
        base_dir=base_dir,
        test_original_dir=originals_dir,
        repair_mode="best_effort",
    )

    assert stats.repair_count == 1
    assert stats.repair_by_source["original"] == 1
    assert "img_a" in stats.repaired_ids

    with zipfile.ZipFile(out_zip, "r") as zf:
        assert zf.read("img_a.jpg") == b"orig-a"
        assert zf.read("img_b.jpg") == b"cand-b"


def test_repair_stats_expose_manifest_fields_for_population(tmp_path: Path):
    expected_names = ["img_a.jpg"]
    primary_dir = tmp_path / "candidate"
    base_dir = tmp_path / "baseline"
    originals_dir = tmp_path / "test_originals"
    out_zip = tmp_path / "out.zip"

    _write_bytes(originals_dir / "img_a.jpg", b"orig-a")
    _write_bytes(base_dir / "img_a.jpg", b"base-a")
    # Missing primary forces fallback to base and produces non-zero repair counters.

    stats = build_zip_from_render_dirs(
        output_zip=out_zip,
        expected_names=expected_names,
        primary_dir=primary_dir,
        base_dir=base_dir,
        test_original_dir=originals_dir,
        repair_mode="best_effort",
    )

    manifest_fragment = {
        "repair_mode": "best_effort",
        "repair_count": int(stats.repair_count),
        "repair_by_source": stats.repair_by_source,
        "repair_failures": stats.repair_failures,
        "repaired_ids_path": "/tmp/repaired_ids.txt",
    }

    assert manifest_fragment["repair_count"] == 1
    assert manifest_fragment["repair_by_source"]["base"] == 1
    assert manifest_fragment["repair_failures"] == []
    assert manifest_fragment["repaired_ids_path"]
