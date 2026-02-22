# AutoHDR Final-Hours Second Brain

Last updated: 2026-02-22 (local)

## Objective
- Maximize final leaderboard score in remaining hackathon time through:
  - parallel Akash model training (3 profiles)
  - continuous heuristic submission loop
  - score-driven iteration, not proxy-only optimization

## Active Plan
1. Launch `rapid_a`, `rapid_b`, `rapid_c` on Akash via `npm run deploy:matrix`.
2. Monitor all DSEQs and terminate unhealthy runs quickly.
3. Run CalibGuard loop every 45-60 minutes:
   - build table
   - produce `safe`, `balanced`, `aggressive` zips
   - upload top 2 candidates
   - analyze returned scored CSV

## Immediate Submission Queue
1. `/Volumes/Love SSD/AutoHDR_Submissions/submission_dim_bucket_microgrid_zero_guard_v1_20260222_030358.zip`
2. `/Volumes/Love SSD/AutoHDR_Submissions/submission_dim_bucket_microgrid_zero_guard_v2_20260222_030614.zip`
3. `/Volumes/Love SSD/AutoHDR_Submissions/submission_dim_bucket_microgrid_zero_guard_v3_20260222_030901.zip`

## Kill Rules
- Kill deployment if:
  - Kaggle auth failure appears (401/403)
  - readiness remains unhealthy for >10 minutes
  - observed/estimated cost breaches per-job cap

## Commands
```bash
# Launch 3-profile matrix
AKASH_MAX_AKT_PER_HOUR_TOTAL=4.0 AKASH_DEPLOYMENT_COUNT=3 npm run deploy:matrix

# Monitor single deployment
npm run monitor -- --dseq <DSEQ> --poll-seconds 90 --cost-cap 1.34

# Build CalibGuard table
python -m backend.scripts.heuristics.build_calibguard_dim_table

# Generate three CalibGuard profiles
python -m backend.scripts.heuristics.heuristic_calibguard_dim --profile safe --artifact-tag calibguard_dim
python -m backend.scripts.heuristics.heuristic_calibguard_dim --profile balanced --artifact-tag calibguard_dim
python -m backend.scripts.heuristics.heuristic_calibguard_dim --profile aggressive --artifact-tag calibguard_dim

# Analyze scored CSV feedback
python -m backend.scripts.heuristics.analyze_scored_submission \
  --score-csv submission.csv \
  --out-json backend/outputs/scored_analysis_latest.json
```

## Notes
- `KAGGLE_API_TOKEN` is valid for `kaggle competitions files/download`.
- Akash deployer now supports token-first auth and preflight validation.
