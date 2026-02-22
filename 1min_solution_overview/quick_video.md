# Quick 1-Minute Video Kit

## Build Order (Fast)

1. Duplicate a 5-slide deck.
2. Drop visuals from the table below.
3. Paste the on-slide text.
4. Read the script once cold, once timed.
5. Record final take (target: 55-65s).

## Build 5 Slides (No Live Demo)

| Slide | Title | Put on slide | Visual |
|---|---|---|---|
| 1 | Opening problem | "Fishbowl real-estate photos: this is what we needed to fix." | `/Users/aaron/Desktop/AutoHDR/1min_solution_overview/assets/figures/fig_05_before_after_example.png` |
| 2 | The overbuild phase | "24-hour sprint: cloud path + notebook path + automation chain + ops governance." Optional prop: scroll `/Users/aaron/Desktop/AutoHDR/plan.md`. | `/Users/aaron/Desktop/AutoHDR/1min_solution_overview/assets/figures/fig_03_real_vs_probe_counts.png` |
| 3 | The humbling results | `0.00` deep-learning hard reject, CalibGuard regression, Akash remained secondary. | `/Users/aaron/Desktop/AutoHDR/1min_solution_overview/assets/figures/fig_01_score_progression.png` |
| 4 | What actually won | 7-bucket dimension heuristic + ZIP-backed per-image fusion (`max_per_image_across_inputs`). | `/Users/aaron/Desktop/AutoHDR/1min_solution_overview/assets/figures/fig_02_top_source_mix.png` + `/Users/aaron/Desktop/AutoHDR/1min_solution_overview/assets/figures/fig_04_failsafe8_patch.png` |
| 5 | Close + lesson | Real best `31.63214`; simple, validated method beat unfinished complexity. | `/Users/aaron/Desktop/AutoHDR/1min_solution_overview/assets/figures/fig_03_real_vs_probe_counts.png` |

## 60-Second Script (Comedic, Accuracy-Checked)

Format: `MM:SS-MM:SS -> narration`

`00:00-00:10` -> You know those Zillow photos where the room looks like a fishbowl? That lens distortion was our target.

`00:10-00:24` -> We had a 24-hour sprint, so naturally we overbuilt: Akash deployment scripts, Kaggle notebook automation, bounty-to-Kaggle orchestration, and an ops log that got very long, very fast.

`00:24-00:38` -> Then reality hit. One deep-learning branch produced a `0.00` hard reject, CalibGuard cycle-1 safe regressed to `18.40668`, and Akash stayed secondary without a promoted scored candidate.

`00:38-00:52` -> What worked was simpler science: a 7-bucket dimension-aware heuristic, then ZIP-backed per-image fusion using `max_per_image_across_inputs` with source counts `572`, `178`, `145`, `64`, and `41`. Real lineage moved from `30.14024` to `30.79062`, then tied at `31.63214`.

`00:52-01:00` -> Probes happened, but governance drew the line: only `ZIP -> bounty -> scored CSV -> Kaggle` counts as real. Sometimes the best engineering is knowing when to stop over-engineering.

## Accuracy Notes (Use These Claims)

1. Real ZIP-backed best score: `31.63214` (`2026-02-22 18:23:37 UTC` and `2026-02-22 18:37:46 UTC`).
2. Baseline real score: `30.14024` (`2026-02-22 02:03:51 UTC`).
3. Uplift real score: `30.79062` (`2026-02-22 15:27:32 UTC`).
4. CalibGuard cycle-1 safe regression: `18.40668` (Kaggle row in lineage map).
5. One hard-reject model artifact scored `0.00` (`v11_submission.csv` in ops log).
6. Lineage snapshot at `2026-02-22 19:00:30 UTC`: `19` real rows and `14` probe rows.

## Final Checks

1. Runtime must land between 55 and 65 seconds.
2. Keep probe scores out of "final result" language.
3. Say explicitly that top real ZIP-backed score is `31.63214`.
4. Export as `autohdr_1min_solution_overview_v1.mp4` (1080p MP4).
5. If you need proof lines while presenting, reference:
`/Users/aaron/Desktop/AutoHDR/docs/ops/real_submission_lineage.md`,
`/Users/aaron/Desktop/AutoHDR/docs/ops/log.md`,
`/Users/aaron/Desktop/AutoHDR/backend/scripts/heuristics/heuristic_dim_bucket.py`,
`/Users/aaron/Desktop/AutoHDR/backend/scripts/heuristics/submission_v4_oracle_valid_allzip_20260222_175058_manifest.json`,
`/Users/aaron/Desktop/AutoHDR/backend/outputs/kaggle/submission_v4_oracle_valid_allzip_failsafe8_20260222_122601_manifest.json`.
