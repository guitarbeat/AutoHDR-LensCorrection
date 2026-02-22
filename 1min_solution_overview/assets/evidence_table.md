# Evidence Table

| Claim | Source path |
|---|---|
| Only real chain is `ZIP (1000 JPGs) -> bounty scoring -> scored CSV -> Kaggle submit`. | `/Users/aaron/Desktop/AutoHDR/plan.md` |
| Baseline real `submission_v4.csv` scored `30.14024` at Kaggle UTC `2026-02-22 02:03:51.610000`. | `/Users/aaron/Desktop/AutoHDR/docs/ops/real_submission_lineage.md` |
| Uplift real `submission_v4_oracle_allbest_20260222_145359_rescored.csv` scored `30.79062` at UTC `2026-02-22 15:27:32.923000`. | `/Users/aaron/Desktop/AutoHDR/docs/ops/real_submission_lineage.md` |
| Top real tied score is `31.63214` for `submission_v4_oracle_valid_allzip...` at UTC `2026-02-22 18:23:37.570000` and failsafe8 at UTC `2026-02-22 18:37:46.543000`. | `/Users/aaron/Desktop/AutoHDR/docs/ops/real_submission_lineage.md` |
| Top solution mechanism uses rule `max_per_image_across_inputs`. | `/Users/aaron/Desktop/AutoHDR/backend/outputs/kaggle/real_oracle_variants_20260222_1241_manifest.json` |
| Top solution source counts include `572` (cycle2_t0), `178` (cycle2_aggressive), `145` (cycle1_safe), `64` (mix_zle0), `41` (learned_pos30). | `/Users/aaron/Desktop/AutoHDR/backend/scripts/heuristics/submission_v4_oracle_valid_allzip_20260222_175058_manifest.json` |
| Oracle estimate `predicted_mean` is `31.19849999999999`. | `/Users/aaron/Desktop/AutoHDR/backend/scripts/heuristics/submission_v4_oracle_valid_allzip_20260222_175058_manifest.json` |
| Failsafe8 patch used `replaced_count: 8` IDs from test originals. | `/Users/aaron/Desktop/AutoHDR/backend/outputs/kaggle/submission_v4_oracle_valid_allzip_failsafe8_20260222_122601_manifest.json` |
| CalibGuard cycle1 safe underperformed with real lineage score `18.40668`. | `/Users/aaron/Desktop/AutoHDR/docs/ops/real_submission_lineage.md` |
| CalibGuard cycle2 balanced underperformed with real lineage score `27.70382`. | `/Users/aaron/Desktop/AutoHDR/docs/ops/real_submission_lineage.md` |
| Probe rows include `submission_constant100...` with `100.00000` and `submission_constant200...` with `0.00000` and are marked non-lineage. | `/Users/aaron/Desktop/AutoHDR/docs/ops/real_submission_lineage.md` |
| Invalidation status remained pending at `2026-02-22 19:00:30 UTC`. | `/Users/aaron/Desktop/AutoHDR/docs/ops/invalidation_followup_20260222_1828.md` |
| Brown-Conrady undistortion with bucket/dimension-aware coefficients is the geometric core in the heuristic pipeline. | `/Users/aaron/Desktop/AutoHDR/backend/scripts/heuristics/heuristic_dim_bucket.py` |
| Submission guardrails enforce lineage and consistency checks before Kaggle submit. | `/Users/aaron/Desktop/AutoHDR/backend/scripts/bounty_to_kaggle_submit.py` |

- [ ] All numeric claims in `script_60s.md` and `slides_outline.md` are traced to source paths above.
