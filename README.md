# AutoHDR

Lens-correction competition workspace (training, heuristics, Kaggle/bounty submission ops).

## Canonical Submission Pipeline

Only this flow is considered a real submission lineage:

`images ZIP (1000 JPGs) -> bounty scoring -> *_scored.csv -> Kaggle submit`

Do not treat score-space probes (`submission_constant*`, `submission_oracle_scores_envelope*`) as real model/image outputs.

## Canonical Docs

- Closeout plan: `/Users/aaron/Desktop/AutoHDR/plan.md`
- Operations log (append-only): `/Users/aaron/Desktop/AutoHDR/docs/ops/log.md`
- Real/probe lineage map: `/Users/aaron/Desktop/AutoHDR/docs/ops/real_submission_lineage.md`

## Key Commands

Run from repo root: `/Users/aaron/Desktop/AutoHDR`

```bash
# Score an image ZIP through bounty (real pipeline step)
python -m backend.scripts.bounty_submit \
  --zip-file "/Volumes/Love SSD/AutoHDR_Submissions/<submission>.zip" \
  --out-csv "/Users/aaron/Desktop/AutoHDR/backend/outputs/bounty/<submission>_scored.csv" \
  --out-summary-json "/Users/aaron/Desktop/AutoHDR/backend/outputs/bounty/<submission>_summary.json"

# Score + submit to Kaggle (now guarded by strict zip/csv lineage checks by default)
python -m backend.scripts.bounty_to_kaggle_submit \
  --zip-file "/Volumes/Love SSD/AutoHDR_Submissions/<submission>.zip" \
  --out-csv "/Users/aaron/Desktop/AutoHDR/backend/outputs/kaggle/<submission>_scored.csv" \
  --out-summary-json "/Users/aaron/Desktop/AutoHDR/backend/outputs/kaggle/<submission>_summary.json" \
  --message "real zip-backed run"
```

## Repo Hygiene

Generated competition outputs are ignored from git via `.gitignore` (`backend/outputs/` and generated heuristic manifests/analysis files).
