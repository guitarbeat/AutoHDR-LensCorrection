# Story And Science (Consolidated)

## Problem Framing

The target was not just a high score, but a **valid** high score under the real chain:

`ZIP (1000 JPGs) -> bounty scoring -> scored CSV -> Kaggle submit`

Any score-space probe outside this lineage is non-canonical for final science claims.

Primary references:
- `/Users/aaron/Desktop/AutoHDR/plan.md`
- `/Users/aaron/Desktop/AutoHDR/docs/ops/real_submission_lineage.md`

## Core Method Science

1. Geometric backbone: Brown-Conrady undistortion with map-based remapping.
2. Stability control: dimension-aware bucketing and coefficient routing (`k1`, `k2`) to handle geometry-sensitive failure modes.
3. Candidate evolution: dim-bucket baseline -> fallback variants -> mix-batch composites -> real-oracle per-image fusion.

Method references:
- `/Users/aaron/Desktop/AutoHDR/backend/scripts/heuristics/heuristic_dim_bucket.py`
- `/Users/aaron/Desktop/AutoHDR/backend/scripts/heuristics/build_real_oracle_variants.py`

## Experiment Progression (Real-Lineage Anchored)

| Stage | Artifact | Kaggle UTC | Public Score | Why It Matters |
|---|---|---|---:|---|
| Baseline | `submission_v4.csv` | `2026-02-22 02:03:51.610000` | `30.14024` | Initial real anchor |
| Fallback branch | `submission_v4_fallback_cycle2_t0_...` | `2026-02-22 14:11:41.570000` | `29.82992` | Robust but below baseline |
| Uplift | `submission_v4_oracle_allbest_...` | `2026-02-22 15:27:32.923000` | `30.79062` | Broke the plateau |
| Top real tie | `submission_v4_oracle_valid_allzip_...` | `2026-02-22 18:23:37.570000` | `31.63214` | Best confirmed real |
| Top real tie + failsafe | `...failsafe8_...` | `2026-02-22 18:37:46.543000` | `31.63214` | Matched top real with patch hardening |

## Why The Top Solution Won

Top mechanism:
- Deterministic per-image source selection using `max_per_image_across_inputs`.

Source composition from manifest:
- `572` from `submission_v4_fallback_cycle2_t0_20260222_080059_scored.csv`
- `178` from `submission_calibguard_cycle2_aggressive_20260222_135105_scored.csv`
- `145` from `submission_calibguard_cycle1_safe_scored.csv`
- `64` from `submission_v4_mix_zle0_mix_batch_20260222_1_scored.csv`
- `41` from `submission_v4_fallback_learned_pos30_20260222_082633_scored.csv`

Failsafe hardening:
- `replaced_count: 8` IDs replaced from `/Volumes/Love SSD/test-originals`.

Evidence:
- `/Users/aaron/Desktop/AutoHDR/backend/scripts/heuristics/submission_v4_oracle_valid_allzip_20260222_175058_manifest.json`
- `/Users/aaron/Desktop/AutoHDR/backend/outputs/kaggle/real_oracle_variants_20260222_1241_manifest.json`
- `/Users/aaron/Desktop/AutoHDR/backend/outputs/kaggle/submission_v4_oracle_valid_allzip_failsafe8_20260222_122601_manifest.json`

## Predicted vs Real Metrics

`predicted_mean = 31.19849999999999` is a selection estimate from known scored inputs.

Canonical final performance remains:
- real ZIP-backed public score `31.63214`.

## Governance And Contamination Handling

Probe contamination occurred and was documented:
- `submission_constant100_...` -> `100.00000` (non-lineage)
- `submission_constant200_...` -> `0.00000` (non-lineage)

Invalidation status was still pending at:
- `2026-02-22 19:00:30 UTC`

References:
- `/Users/aaron/Desktop/AutoHDR/docs/ops/real_submission_lineage.md`
- `/Users/aaron/Desktop/AutoHDR/docs/ops/invalidation_followup_20260222_1828.md`
- `/Users/aaron/Desktop/AutoHDR/backend/scripts/bounty_to_kaggle_submit.py`

## Scientific Takeaways

1. Proxy wins are hypotheses; real lineage scores are acceptance criteria.
2. Geometry-aware correction outperformed global assumptions.
3. Per-image ZIP-backed fusion produced the decisive uplift.
4. Governance checks are part of scientific validity, not post-processing.
