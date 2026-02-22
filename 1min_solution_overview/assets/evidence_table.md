# Evidence Table

| Claim (used in slides/script) | Source path |
|---|---|
| Canonical real chain is `ZIP (1000 JPGs) -> bounty scoring -> scored CSV -> Kaggle submit`. | `/Users/aaron/Desktop/AutoHDR/plan.md` |
| Baseline real `submission_v4.csv` scored `30.14024` at UTC `2026-02-22 02:03:51.610000`. | `/Users/aaron/Desktop/AutoHDR/docs/ops/real_submission_lineage.md` |
| Uplift real `submission_v4_oracle_allbest_20260222_145359_rescored.csv` scored `30.79062`. | `/Users/aaron/Desktop/AutoHDR/docs/ops/real_submission_lineage.md` |
| Top real tied score is `31.63214` (valid allzip + failsafe8 lineage rows). | `/Users/aaron/Desktop/AutoHDR/docs/ops/real_submission_lineage.md` |
| Top fusion rule is `max_per_image_across_inputs`. | `/Users/aaron/Desktop/AutoHDR/backend/outputs/kaggle/real_oracle_variants_20260222_1241_manifest.json` |
| Top source counts are `572`, `178`, `145`, `64`, `41`. | `/Users/aaron/Desktop/AutoHDR/backend/scripts/heuristics/submission_v4_oracle_valid_allzip_20260222_175058_manifest.json` |
| Oracle estimate `predicted_mean` is `31.19849999999999`. | `/Users/aaron/Desktop/AutoHDR/backend/scripts/heuristics/submission_v4_oracle_valid_allzip_20260222_175058_manifest.json` |
| Failsafe patch shows `replaced_count: 8`. | `/Users/aaron/Desktop/AutoHDR/backend/outputs/kaggle/submission_v4_oracle_valid_allzip_failsafe8_20260222_122601_manifest.json` |
| Probe contamination examples: `constant100 -> 100.00000`, `constant200 -> 0.00000` (both non-lineage). | `/Users/aaron/Desktop/AutoHDR/docs/ops/real_submission_lineage.md` |
| Invalidation status reference: pending at `2026-02-22 19:00:30 UTC`. | `/Users/aaron/Desktop/AutoHDR/docs/ops/invalidation_followup_20260222_1828.md` |
| Method core is Brown-Conrady with dimension-aware coefficients. | `/Users/aaron/Desktop/AutoHDR/backend/scripts/heuristics/heuristic_dim_bucket.py` |
| Guardrails enforce lineage and score/summary consistency pre-submit. | `/Users/aaron/Desktop/AutoHDR/backend/scripts/bounty_to_kaggle_submit.py` |

- [ ] All numeric claims in `presentation_packet.md` are traced to source paths above.
