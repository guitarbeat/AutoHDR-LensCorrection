# Solution Mechanics (Science-First)

## 1) Geometric Core Method

AutoHDR correction is grounded in Brown-Conrady undistortion, implemented with OpenCV map-based remapping (`k1`, `k2`, `alpha=0`) and tuned for stability by image geometry buckets.

Primary method reference:
- `/Users/aaron/Desktop/AutoHDR/backend/scripts/heuristics/heuristic_dim_bucket.py`

Why this matters:
- Distortion response is not uniform across dimensions.
- Portrait and heavy-crop shapes were empirically riskier.
- Dimension-aware coefficients reduce failure modes that appeared in global settings.

## 2) Candidate Family Evolution

The candidate family evolved in stages:

1. Dimension-bucket baseline and micro-grid tuning.
2. Fallback variants (`cycle2_t0`, `learned_pos30`, `t5`) for robustness.
3. Mix-batch composites over ZIP-backed candidates.
4. Real-oracle fusion variants selecting one source image per `image_id` from scored ZIP-backed inputs.

Relevant implementation and manifests:
- `/Users/aaron/Desktop/AutoHDR/backend/scripts/heuristics/heuristic_dim_bucket.py`
- `/Users/aaron/Desktop/AutoHDR/backend/scripts/heuristics/submission_mix_batch.py`
- `/Users/aaron/Desktop/AutoHDR/backend/scripts/heuristics/build_real_oracle_variants.py`
- `/Users/aaron/Desktop/AutoHDR/backend/outputs/kaggle/real_oracle_variants_20260222_1241_manifest.json`

## 3) How The Top Real Solution Works

Top real ZIP-backed submission family:
- `submission_v4_oracle_valid_allzip_20260222_175058.zip`
- `submission_v4_oracle_valid_allzip_failsafe8_20260222_122601.zip`

Top real public score (tied):
- `31.63214` (Kaggle UTC `2026-02-22 18:23:37.570000` and `2026-02-22 18:37:46.543000`)
- Source: `/Users/aaron/Desktop/AutoHDR/docs/ops/real_submission_lineage.md`

Mechanism details from the oracle manifest:
- Rule: `max_per_image_across_inputs`
- `predicted_mean`: `31.19849999999999`
- Per-image source counts:
- `572` from `submission_v4_fallback_cycle2_t0_20260222_080059_scored.csv`
- `178` from `submission_calibguard_cycle2_aggressive_20260222_135105_scored.csv`
- `145` from `submission_calibguard_cycle1_safe_scored.csv`
- `64` from `submission_v4_mix_zle0_mix_batch_20260222_1_scored.csv`
- `41` from `submission_v4_fallback_learned_pos30_20260222_082633_scored.csv`

Evidence:
- `/Users/aaron/Desktop/AutoHDR/backend/scripts/heuristics/submission_v4_oracle_valid_allzip_20260222_175058_manifest.json`
- `/Users/aaron/Desktop/AutoHDR/backend/outputs/kaggle/real_oracle_variants_20260222_1241_manifest.json`

## 4) Failsafe8 Patch

The failsafe variant patched `8` image IDs by replacing selected outputs with originals from test images.

Failsafe evidence:
- `/Users/aaron/Desktop/AutoHDR/backend/outputs/kaggle/submission_v4_oracle_valid_allzip_failsafe8_20260222_122601_manifest.json`

Key fields:
- `replaced_count: 8`
- `source_for_replacements: /Volumes/Love SSD/test-originals`

## 5) Predicted Mean vs Real Leaderboard Score

The oracle manifest `predicted_mean` is an internal estimate from known scored sources, not the final competition truth.

Interpretation contract:
1. `predicted_mean` supports candidate selection and ranking.
2. Real performance is only what appears in Kaggle for a ZIP-backed lineage submission.
3. Therefore, `31.63214` (real lineage score) is the canonical top real result, not `31.19849999999999` (predicted oracle mean).

Canonical governance references:
- `/Users/aaron/Desktop/AutoHDR/plan.md`
- `/Users/aaron/Desktop/AutoHDR/docs/ops/real_submission_lineage.md`
