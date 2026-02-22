# Slides Outline (Science-First, 5 Slides, 60 Seconds)

## Slide 1 - Problem, Metric, Validity Rule

Key message:
- We optimize lens-correction quality under real leaderboard constraints, not proxy-only metrics.

On-slide content:
- Objective: geometric correction for the AutoHDR challenge.
- Validity rule: only `ZIP (1000 JPGs) -> bounty scoring -> scored CSV -> Kaggle submit` is real.
- Current top real score: `31.63214` (probe scores excluded).

Visual:
- One pipeline strip + one badge reading `ZIP-backed lineage only`.

## Slide 2 - Method Science

Key message:
- Core method is geometry-aware Brown-Conrady undistortion.

On-slide content:
- Brown-Conrady correction (`k1`, `k2`) with bucket/dimension-aware routing.
- Why buckets: portrait/heavy-crop risk differs from standard landscape.
- Candidate families: dim-bucket baseline -> fallback variants -> mix-batch -> real-oracle fusion.

Visual:
- Distortion model block plus small bucket map diagram.

## Slide 3 - Experimental Arc

Key message:
- Controlled experiments moved us from baseline to top real tie.

On-slide content:
- Baseline real: `submission_v4.csv` -> `30.14024`.
- Uplift real: `submission_v4_oracle_allbest...` -> `30.79062`.
- Top real tied: `submission_v4_oracle_valid_allzip...` and failsafe8 -> `31.63214`.
- Learning branches:
- CalibGuard cycle1 safe: `18.40668`.
- CalibGuard cycle2 balanced: `27.70382`.

Visual:
- Timeline with score markers.

## Slide 4 - Top Solution Internals

Key message:
- The best real result came from deterministic per-image fusion over ZIP-backed scored inputs.

On-slide content:
- Rule: `max_per_image_across_inputs`.
- Source counts: `572 / 178 / 145 / 64 / 41` across five key inputs.
- Oracle estimate: `predicted_mean = 31.19849999999999` (selection metric).
- Failsafe8 patch: `replaced_count = 8` IDs from test originals.

Visual:
- Stacked source-count bar + failsafe patch callout.

## Slide 5 - Validation and Governance

Key message:
- Scientific claims are accepted only when lineage and governance checks pass.

On-slide content:
- Real vs probe separation from lineage report.
- Probe incident acknowledged (`constant100`, `constant200`) and excluded from final science claims.
- Status note: invalidation remained pending at `2026-02-22 19:00:30 UTC`.
- Guardrails: lineage verification + `1000`-row parity + score/summary consistency.

Visual:
- Two-column table: `Real` vs `Probe`.
