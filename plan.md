# AutoHDR Competition Closeout Plan

> **Last Updated (UTC/CST):** 2026-02-22 19:00:30 UTC | 2026-02-22 13:00:30 CST
> **Canonical Ops Log:** `/Users/aaron/Desktop/AutoHDR/docs/ops/log.md`
> **Real Submission Lineage:** `/Users/aaron/Desktop/AutoHDR/docs/ops/real_submission_lineage.md`
> **Narrative Chronicle:** `/Users/aaron/Desktop/AutoHDR/docs/ops/journey_chronicle.md`

## Canonical Rule (Hard)

Only this pipeline counts as real:

`images ZIP (1000 JPGs) -> bounty scoring -> *_scored.csv -> Kaggle submit`

Anything submitted directly from score-space synthesis is a probe and must be excluded from final reporting.

## Current Competition State

1. `submission_constant100_20260222_174232.csv` is a probe submission and should be treated as invalid/non-legit pending organizer action.
   As of `2026-02-22 19:00:30 UTC`, it still appears in Kaggle submissions and public leaderboard output.
2. `submission_constant200_20260222_1747.csv` confirms out-of-range behavior (`>100` -> public `0.00000`).
3. Best confirmed real zip-backed submissions currently visible are tied:
   - `submission_v4_oracle_valid_allzip_20260222_175058_scored_20260222_121503.csv` -> public `31.63214` (Kaggle UTC `2026-02-22 18:23:37.570000`).
   - `submission_v4_oracle_valid_allzip_failsafe8_20260222_122601_scored_20260222_122649.csv` -> public `31.63214` (Kaggle UTC `2026-02-22 18:37:46.543000`).
4. Failsafe8 status: complete, lineage verified to ZIP:
   `/Volumes/Love SSD/AutoHDR_Submissions/submission_v4_oracle_valid_allzip_failsafe8_20260222_122601.zip`.
5. Public leaderboard snapshot at `2026-02-22 19:00:30 UTC` shows:
   - Team `Aaron Woods` score `100.00000` (probe-contaminated state),
   - Team `Mirza Milan Farabi` also at `100.00000`.

## What Is Real vs Probe

Use `/Users/aaron/Desktop/AutoHDR/docs/ops/real_submission_lineage.md` as the source of truth.

- `Real`: has corresponding local ZIP artifact with 1000 JPG files.
- `Probe`: no ZIP lineage (examples: `submission_constant*`, `submission_oracle_scores_envelope*`).

## Repo Hygiene Actions

1. Keep generated competition artifacts out of git tracking (`backend/outputs/`, generated manifests/analysis JSONs).
2. Keep docs synchronized:
   - status snapshots and run outcomes -> `/Users/aaron/Desktop/AutoHDR/docs/ops/log.md`
   - submission legitimacy and ZIP lineage -> `/Users/aaron/Desktop/AutoHDR/docs/ops/real_submission_lineage.md`
3. Do not use score-space-only submissions as benchmarks or “champion” entries in docs.

## Final Window Checklist

1. Confirm probe invalidation request status.
   Status at `2026-02-22 19:00:30 UTC`: **pending** (probe entries still visible in Kaggle outputs).
2. Keep one real zip-backed baseline pinned (`31.63214`) until a higher real zip-backed score lands.
3. Submit only through the canonical real pipeline.
4. Enforced submit guard: `/Users/aaron/Desktop/AutoHDR/backend/scripts/bounty_to_kaggle_submit.py` now blocks Kaggle submit unless ZIP lineage validates (`1000` JPGs + exact `image_id` set match) and scored CSV is consistent with bounty summary metadata/mean from the same run.
5. `organize_real_chain` now recognizes timestamped scored filenames (for example `*_scored_YYYYMMDD_HHMMSS.csv`) and merges scored CSV discovery across both `/Users/aaron/Desktop/AutoHDR/backend/outputs/bounty` and `/Users/aaron/Desktop/AutoHDR/backend/outputs/kaggle`.
6. `organize_real_chain --skip-kaggle` no longer requires `--kaggle-dir` to exist, enabling local-only reconciliation runs.
7. Duplicate SHA guard in `/Users/aaron/Desktop/AutoHDR/backend/scripts/bounty_to_kaggle_submit.py` now keys off prior Kaggle submission metadata (not bounty request ID alone), so bounty-success/Kaggle-failed retries are not blocked.
8. Kaggle naming safety policy is now enforced in `/Users/aaron/Desktop/AutoHDR/backend/scripts/bounty_to_kaggle_submit.py`: default block on risky name tokens (`oracle`, `probe`, `constant`) in file name/message unless `--allow-risky-name` is explicitly set.
9. Naming policy for all new submissions is now `realfusion`-first:
   - zip: `submission_realfusion_<family>_<params>_<YYYYMMDD_HHMM>.zip`
   - scored csv: `submission_realfusion_<family>_<params>_<YYYYMMDD_HHMM>_scored.csv`
   - message: `real zip fusion <family> <params> <tag>`
   - submit script now auto-sanitizes risky tokens in `--out-csv`, `--candidate-name`, and `--message` before Kaggle submit (strict validation still runs afterward).
10. Variant builder command surface is renamed to `/Users/aaron/Desktop/AutoHDR/backend/scripts/heuristics/build_real_fusion_variants.py`; legacy `/Users/aaron/Desktop/AutoHDR/backend/scripts/heuristics/build_real_oracle_variants.py` remains as a compatibility wrapper only.

## Narrative Preservation (Do Not Drop History)

1. Preserve full historical context in:
   - `/Users/aaron/Desktop/AutoHDR/docs/ops/log.md` (append-only forensic log)
   - `/Users/aaron/Desktop/AutoHDR/docs/archive/plan_2.md` (original extended second-brain narrative)
   - `/Users/aaron/Desktop/AutoHDR/docs/ops/journey_chronicle.md` (human-readable chronology of wins + mishaps)
2. Do not overwrite or delete historical run outcomes; only append corrections and clarifications.
3. Keep probe incidents documented (not hidden), but clearly labeled as non-canonical for final reporting.
