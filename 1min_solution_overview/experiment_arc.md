# Experiment Arc (What We Tried, What Won, Why)

## Chronological Sequence (Real Lineage Anchored)

1. **Baseline real anchor**
- Kaggle UTC `2026-02-22 02:03:51.610000`
- File: `submission_v4.csv`
- Public score: `30.14024`
- Source: `/Users/aaron/Desktop/AutoHDR/docs/ops/real_submission_lineage.md`

2. **Fallback family stabilization**
- `submission_v4_fallback_t5_20260222_074756_scored.csv` -> `29.69816`
- `submission_v4_fallback_cycle2_t0_20260222_080059_scored.csv` -> `29.82992`
- `submission_v4_fallback_learned_pos30_20260222_082633_scored.csv` -> `30.03266`
- Interpretation: robust but not enough uplift versus baseline.

3. **Oracle-validated uplift**
- Kaggle UTC `2026-02-22 15:27:32.923000`
- File: `submission_v4_oracle_allbest_20260222_145359_rescored.csv`
- Public score: `30.79062`
- Interpretation: deterministic composition over scored ZIP-backed artifacts broke the earlier plateau.

4. **Top real tied state**
- `submission_v4_oracle_valid_allzip_20260222_175058_scored_20260222_121503.csv` -> `31.63214` (UTC `2026-02-22 18:23:37.570000`)
- `submission_v4_oracle_valid_allzip_failsafe8_20260222_122601_scored_20260222_122649.csv` -> `31.63214` (UTC `2026-02-22 18:37:46.543000`)
- Source: `/Users/aaron/Desktop/AutoHDR/docs/ops/real_submission_lineage.md`

## Failed / Learning Branches

1. **Proxy overtrust regressions**
- Multiple proxy-positive directions regressed under real scoring.
- Source: `/Users/aaron/Desktop/AutoHDR/docs/ops/journey_chronicle.md`

2. **CalibGuard underperformance versus top real family**
- `submission_calibguard_cycle1_safe_20260222_102221.zip` lineage row via `submission.csv` -> `18.40668`
- `submission_calibguard_cycle2_balanced_20260222_134848.zip` lineage row via `submission.csv` -> `27.70382`
- Interpretation: useful signals for robustness diagnostics, not final champion path.

3. **Probe contamination event**
- `submission_constant100_20260222_174232.csv` -> `100.00000` (probe, no ZIP lineage)
- `submission_constant200_20260222_1747.csv` -> `0.00000` (probe, no ZIP lineage)
- Invalidation status remained pending at `2026-02-22 19:00:30 UTC`.
- Sources:
- `/Users/aaron/Desktop/AutoHDR/docs/ops/real_submission_lineage.md`
- `/Users/aaron/Desktop/AutoHDR/docs/ops/invalidation_followup_20260222_1828.md`

## Scientific Lessons Learned

1. Local proxy improvements are hypotheses; only real ZIP-backed chain scores are acceptance criteria.
2. Distortion correction quality is geometry-sensitive; dimension-aware parameterization outperformed global assumptions.
3. Per-image source selection across validated scored ZIPs produced larger gains than single-candidate tuning alone.
4. Governance controls are part of scientific validity: non-lineage probes cannot be used as final model evidence.
