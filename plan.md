# AutoHDR Live Dashboard

> **Last Updated (UTC/CST):** 2026-02-22 14:28:02 UTC | 2026-02-22 08:28:02 CST
> **Hackathon Window (CST):** 2026-02-21 17:00 CST -> 2026-02-22 17:00 CST
> **Append-Only Log:** `/Users/aaron/Desktop/AutoHDR/docs/ops/log.md`

## Status Snapshot

- **[2026-02-22 14:28:02 UTC | 2026-02-22 08:28:02 CST]** Champion Kaggle score: `submission_v4.csv` public `30.14024`.
- **[2026-02-22 14:28:02 UTC | 2026-02-22 08:28:02 CST]** Scoring baseline for current queue: `/Users/aaron/Desktop/AutoHDR/backend/outputs/bounty/submission_v4_fallback_cycle2_t0_20260222_080059_scored.csv` mean `29.36779`, zero-rate `19.40%`.
- **[2026-02-22 14:28:02 UTC | 2026-02-22 08:28:02 CST]** Mix priors: `zero_id` beats baseline on `57` IDs (`near_standard_short=32`, `standard=13`), predicted mean uplift `+0.09828` for `mix_zpos_t5plus` and `+0.09643` for `mix_zpos`.
- **[2026-02-22 14:28:02 UTC | 2026-02-22 08:28:02 CST]** Run budget: max `4` scored experiments before automatic fallback to champion lineage.
- **[2026-02-22 14:28:02 UTC | 2026-02-22 08:28:02 CST]** Parallel track policy: keep heuristic + CalibGuard tracks, but never exceed 4-run cap.

## Competitive Context

- Snapshot timestamp: **2026-02-22 14:15:55 UTC | 2026-02-22 08:15:55 CST**.
- Team `alwoods` rank: **25/69** with score **30.14024**.
- Gap to rank 24 (`30.85952`): **0.71928**.
- Margin over rank 26 (`29.50104`): **0.63920**.
- Context refresh cadence: hourly during active submission window.

## Next Actions (Priority)

1. **Run 1: `mix_zpos_t5plus`** (ETA 35 min). Hypothesis: patching proven winners from `zero_id` plus 2 `t5` overrides lifts mean with minimal bucket churn. Stop if Tier A catastrophic trigger fires.
2. **Run 2: `mix_zpos`** (ETA 30 min). Hypothesis: same gain channel as Run 1 with lower interaction risk. Stop if Run 1 already clears champion replacement gate.
3. **Run 3: `calibguard_cycle2_balanced` score** (ETA 20 min scoring + queue). Hypothesis: dimension guardrails improve against cycle1-safe regression.
4. **Run 4: `calibguard_cycle2_aggressive` score** (ETA 20 min scoring + queue). Hypothesis: aggressive routing recovers mean if guardrails prevent heavy failures.
5. **Fallback rule:** if no run clears Tier B by T-60 minutes (2026-02-22 16:00 CST / 22:00 UTC), submit champion lineage and stop experimentation.

## Decision Gates v2

- **Tier A (operational promotion gate, pass/fail for continuing a candidate):**
1. Novel artifact required (`sha256` differs from any previously scored zip/csv lineage).
2. Pre-score run card required: hypothesis, targeted buckets, expected delta range, stop condition.
3. Candidate is operationally viable if `mean >= current_best - 0.50` **or** `mean >= baseline + 0.10`.
4. Candidate is catastrophic-reject if any: `mean < 27.0`, `zero-rate > 25%`, `hard_fails > 12`.
- **Tier B (champion replacement gate):**
1. Replace champion if `mean >= champion + 0.20`.
2. Or replace when `mean >= champion - 0.10` **and** zero-rate improves by `>= 2.0` points.
- **Aspirational targets (non-blocking):**
1. Overall zero-rate `<= 12%`.
2. `moderate_crop` and `heavy_crop` trend upward over successive runs.

## Quick Commands

```bash
# Build mix-batch candidates (promoted workflow)
python -m backend.scripts.heuristics.submission_mix_batch \
  --base-zip "/Volumes/Love SSD/AutoHDR_Submissions/submission_v4_fallback_cycle2_t0_20260222_080059.zip" \
  --zero-zip "/Volumes/Love SSD/AutoHDR_Submissions/submission_v4_zero_id_fallback_20260222_073515.zip" \
  --t5-zip "/Volumes/Love SSD/AutoHDR_Submissions/submission_v4_fallback_t5_20260222_074756.zip" \
  --base-score-csv "/Users/aaron/Desktop/AutoHDR/backend/outputs/bounty/submission_v4_fallback_cycle2_t0_20260222_080059_scored.csv" \
  --zero-score-csv "/Users/aaron/Desktop/AutoHDR/backend/outputs/bounty/submission_v4_zero_id_fallback_20260222_073515_scored.csv" \
  --t5-score-csv "/Users/aaron/Desktop/AutoHDR/backend/outputs/bounty/submission_v4_fallback_t5_20260222_074756_scored.csv" \
  --test-original-dir "/Volumes/Love SSD/test-originals" \
  --output-dir "/Volumes/Love SSD/AutoHDR_Submissions" \
  --tag "mix_batch_live" \
  --manifest-out "/Users/aaron/Desktop/AutoHDR/backend/scripts/heuristics/submission_mix_batch_manifest_live.json"

# Score one artifact without Kaggle submit
python -m backend.scripts.bounty_to_kaggle_submit --zip-file "<zip>" --skip-kaggle-submit \
  --out-csv "/Users/aaron/Desktop/AutoHDR/backend/outputs/bounty/<name>_scored.csv" \
  --out-summary-json "/Users/aaron/Desktop/AutoHDR/backend/outputs/bounty/<name>_summary.json"
```

## Active Experiment Board

| Run | Artifact | Hypothesis | Expected Delta vs 29.36779 | Stop Condition | Status |
|---|---|---|---:|---|---|
| 1 | `mix_zpos_t5plus` | winning ID patch set + 2 t5 overrides | `+0.05` to `+0.15` | Tier A catastrophic fail | Pending |
| 2 | `mix_zpos` | winning ID patch set only | `+0.05` to `+0.15` | Run 1 already clears Tier B | Pending |
| 3 | `calibguard_cycle2_balanced` | guarded exact-dim routing beats cycle1-safe | `>= +0.10` | Tier A catastrophic fail | Pending |
| 4 | `calibguard_cycle2_aggressive` | aggressive routing improves mean if guardrails hold | `>= +0.10` | Tier A catastrophic fail | Pending |
