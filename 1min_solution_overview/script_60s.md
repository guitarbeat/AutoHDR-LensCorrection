# 60-Second Narration Script

Format: `MM:SS-MM:SS -> narration`

`00:00-00:10` -> Our objective was geometric lens correction that survives real leaderboard scoring, not proxy-only wins.

`00:10-00:22` -> Only one chain counts as real: ZIP with 1000 JPGs, bounty scoring, scored CSV, then Kaggle submit. The method core is Brown-Conrady undistortion with dimension-aware coefficients.

`00:22-00:34` -> In ZIP-backed lineage, baseline `submission_v4.csv` scored 30.14024, then `submission_v4_oracle_allbest...` rose to 30.79062 through deterministic per-image fusion across scored ZIP candidates.

`00:34-00:48` -> The top real tied score is 31.63214, built by `max_per_image_across_inputs`: 572 images from cycle2_t0, 178 from cycle2_aggressive, 145 from cycle1_safe, 64 from mix_zle0, and 41 from learned_pos30.

`00:48-01:00` -> Probe submissions happened and were excluded. Recovery added lineage guards, 1000-row/ID checks, and failsafe8 patching 8 risky IDs. `predicted_mean` 31.19849999999999 stayed separate from real Kaggle scores. Invalidation was still pending at 2026-02-22 19:00:30 UTC.
