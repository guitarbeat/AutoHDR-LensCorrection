# Shot List (Evidence Visuals)

## Shot 1 - Canonical Real Pipeline Rule

Source:
- `/Users/aaron/Desktop/AutoHDR/plan.md`

Capture:
- Section containing:
  `images ZIP (1000 JPGs) -> bounty scoring -> *_scored.csv -> Kaggle submit`

Usage:
- Slide 1 and Slide 5.

## Shot 2 - Real vs Probe Lineage Tables

Source:
- `/Users/aaron/Desktop/AutoHDR/docs/ops/real_submission_lineage.md`

Capture:
- Visible rows from `Real (Select These)` and `Probe / Non-Lineage`.

Usage:
- Slide 5.

## Shot 3 - Top Solution Manifest Fields

Sources:
- `/Users/aaron/Desktop/AutoHDR/backend/scripts/heuristics/submission_v4_oracle_valid_allzip_20260222_175058_manifest.json`
- `/Users/aaron/Desktop/AutoHDR/backend/outputs/kaggle/real_oracle_variants_20260222_1241_manifest.json`

Capture:
- `predicted_mean`
- `source_counts`
- output ZIP path and SHA (`outputs.zip`, `outputs.sha256`)

Usage:
- Slide 4.

## Shot 4 - Failsafe8 Patch Evidence

Source:
- `/Users/aaron/Desktop/AutoHDR/backend/outputs/kaggle/submission_v4_oracle_valid_allzip_failsafe8_20260222_122601_manifest.json`

Capture:
- `replaced_count: 8`
- `replaced_ids`
- `source_for_replacements`

Usage:
- Slide 4.

## Shot 5 - Experiment Arc Timeline Evidence

Sources:
- `/Users/aaron/Desktop/AutoHDR/docs/ops/real_submission_lineage.md`
- `/Users/aaron/Desktop/AutoHDR/docs/ops/journey_chronicle.md`

Capture:
- Baseline row (`30.14024`).
- Uplift row (`30.79062`).
- Top tied real rows (`31.63214`).

Usage:
- Slide 3.
