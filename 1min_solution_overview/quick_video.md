# Quick 1-Minute Video Kit

## Build 5 Slides (No Live Demo)

| Slide | Title | Put on slide | Visual |
|---|---|---|---|
| 1 | Problem + rule | "Lens correction under competition constraints" and `ZIP -> bounty -> scored CSV -> Kaggle` | `/Users/aaron/Desktop/AutoHDR/1min_solution_overview/assets/figures/fig_01_score_progression.png` |
| 2 | Method science | Brown-Conrady undistortion + dimension-aware coefficient routing | `/Users/aaron/Desktop/AutoHDR/1min_solution_overview/assets/figures/fig_05_before_after_example.png` |
| 3 | What we tried | Baseline `30.14024` -> uplift `30.79062` -> top real tie `31.63214` | `/Users/aaron/Desktop/AutoHDR/1min_solution_overview/assets/figures/fig_01_score_progression.png` |
| 4 | Why top solution won | `max_per_image_across_inputs`, source mix `572/178/145/64/41`, failsafe8 patch `8` | `/Users/aaron/Desktop/AutoHDR/1min_solution_overview/assets/figures/fig_02_top_source_mix.png` and `/Users/aaron/Desktop/AutoHDR/1min_solution_overview/assets/figures/fig_04_failsafe8_patch.png` |
| 5 | Governance + recovery | Probe runs excluded; real claims require lineage + 1000-row ID integrity checks | `/Users/aaron/Desktop/AutoHDR/1min_solution_overview/assets/figures/fig_03_real_vs_probe_counts.png` |

## 60-Second Script

Format: `MM:SS-MM:SS -> narration`

`00:00-00:10` -> Our goal was to maximize AutoHDR lens-correction quality under competition pressure, with one hard rule: only full ZIP-backed lineage submissions count as real evidence.

`00:10-00:22` -> Method core: Brown-Conrady undistortion, then dimension-aware coefficient routing, because portraits, vertical crops, and extreme aspect ratios break when one global parameter set is reused.

`00:22-00:35` -> What we tried: baseline `submission_v4.csv` at `30.14024`, conservative fallback variants that were robust but lower, then oracle-allbest fusion at `30.79062`, which finally broke the stagnation plateau.

`00:35-00:48` -> Top real result tied at `31.63214` by selecting each image from the best ZIP-backed source using `max_per_image_across_inputs`, with mix counts `572`, `178`, `145`, `64`, and `41`.

`00:48-01:00` -> Hangup moment: I accidentally speed-ran leaderboard anxiety when probe submissions happened. Recovery was strict lineage plus `1000`-row ID checks, summary consistency checks, duplicate-SHA guardrails, and failsafe8 patching `8` risky IDs. `predicted_mean` `31.19849999999999` stayed estimate only.

## Final Checks

1. Runtime must land between 55 and 65 seconds.
2. Keep probe scores out of "final result" language.
3. Say explicitly that top real ZIP-backed score is `31.63214`.
4. Export as `autohdr_1min_solution_overview_v1.mp4` (1080p MP4).
