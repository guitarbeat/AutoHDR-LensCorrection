# AutoHDR Execution Runbook

> **Last Updated (CST):** February 22, 2026 7:56 AM CST  
> **Hackathon Window:** February 21, 2026 5:00 PM CST → February 22, 2026 5:00 PM CST  
> **Competition:** [Kaggle — Automatic Lens Correction](https://www.kaggle.com/competitions/automatic-lens-correction)  
> **Scoring Portal:** [bounty.autohdr.com](https://bounty.autohdr.com/)

---

## 1. Status Snapshot

- **Best scored heuristic submission:** `submission_dim_bucket_microgrid.csv` *(legacy: `submission_v4.csv`)* with mean score **29.64**
- **Current heuristic script:** `backend/scripts/heuristics/heuristic_dim_bucket.py`
- **Current experimental method:** `CalibGuard-Dim` via `backend/scripts/heuristics/heuristic_calibguard_dim.py`
- **Parallel experimental notebook:** `backend/scripts/kaggle/Train-Hybrid-CalibGuard.ipynb` *(guarded residual + geometry fallback; not canonical promotion path yet)*
- **Single source of truth doc:** `plan.md` *(all operational updates and decisions must be recorded here)*
- **Latest scored heuristic run:** `submission_calibguard_cycle1_safe_scored.csv` (February 22, 2026) mean **18.1704** *(regression; delta vs best: -11.47)*
- **Latest guardrail batch (tab exports, February 22, 2026 ~3:33 AM CST):** best mean **24.9839** *(still regression; no promotion)*
- **Latest scored model run:** `v11_submission.csv` (February 22, 2026) mean **0.0000** *(hard failure; 100% zeros)*
- **Latest bounty automation request:** `1031b2da-202c-465a-9272-47fdf9fd86a7` *(COMPLETE; 1000 scored, 0 missing, 9 hard fails, portal avg_score=0.1817 => CSV mean=18.1704)*
- **Canonical deep-learning run path:** Kaggle notebook via MCP (`run_notebook_mcp.py`)
- **Local deep-learning scripts:** `train.py`, `inference.py`, `evaluate_model.py`
- **Dataset root (default):** `/Volumes/Love SSD`

---

## 1.1 Totals Snapshot (Quick Answer)

As of **February 22, 2026 4:36 AM CST**:

1. **Total scored submissions:** `12`
2. **Estimated unique scored behaviors:** `11` (`safezero_v1` and `zero_guard_v1` were duplicate behavior with identical outputs)
3. **Untested (pending score):** `submission_calibguard_dim_<profile>_<timestamp>.csv`
4. **Current best scored run:** `submission_dim_bucket_microgrid.csv` at `29.64`
5. **Latest scored run:** `submission_calibguard_cycle1_safe_scored.csv` at `18.1704`
6. **Latest automated scoring artifact pair:** `/Users/aaron/Desktop/AutoHDR/backend/outputs/bounty/submission_calibguard_cycle1_safe_scored.csv` + `/Users/aaron/Desktop/AutoHDR/backend/outputs/bounty/submission_calibguard_cycle1_safe_summary.json`

## 1.2 Version Name Translator

Use this mapping when talking to agents to avoid `v1/v2/v3` ambiguity:

| Legacy shorthand | Canonical run name | Technique | Outcome |
|---|---|---|---|
| `v1` *(legacy baseline shorthand)* | `submission_global_grid.csv` *(legacy file: `submission_heuristic.csv`)* | Global coefficients (`k1=-0.17`, `k2=0.35`) | 24.00 (baseline) |
| `v2` | `submission_hough_proxy.csv` *(legacy file: `submission_v2.csv`)* | Per-image Hough proxy selection | 3.45 (rejected) |
| `v3` | `submission_laplacian_proxy.csv` *(legacy file: `submission_v3.csv`)* | Per-image Laplacian proxy selection | 21.21 (rejected) |
| `v4` | `submission_dim_bucket_microgrid.csv` *(legacy file: `submission_v4.csv`)* | Dimension-bucket micro-grid coefficients | **29.64 (current best)** |
| `v5` *(working label only)* | `CalibGuard-Dim` family | Exact-dimension + guardrail routing | Active iteration |

Recent guardrail variants are separate from `v1-v5` shorthand:

- `safezero_v1`, `zero_guard_v1_20260222_030358`, `zero_guard_v2_20260222_030614`, `zero_guard_v3_20260222_030901`.

---

## 2. Objective

Build a geometrically-correct lens distortion pipeline for wide-angle real-estate images and maximize bounty score under hackathon time constraints.

- Prioritize methods that consistently improve real bounty scores.
- Avoid proxy-only optimization loops unless tied to validated score correlation.
- Keep execution reproducible with explicit commands and artifact paths.

### 2.1 Documentation Policy (Canonical)

This repository now uses one canonical operations document:

1. Canonical runbook: `/Users/aaron/Desktop/AutoHDR/plan.md`
2. Deprecated mirrors: `/Users/aaron/Desktop/AutoHDR/docs/ops/research_log.md` and `/Users/aaron/Desktop/AutoHDR/docs/ops/agent_notes.md`
3. Agent handoff rule:
Always reference this file (`plan.md`) first for status, commands, history, gates, and next decisions.

---

## 3. Dataset And Paths

### 3.1 Expected dataset layout

```text
AUTOHDR_DATA_ROOT/
├── lens-correction-train-cleaned/
│   ├── *_original.jpg
│   └── *_generated.jpg
└── test-originals/
    └── *.jpg
```

### 3.2 Environment variables (source of truth)

```bash
AUTOHDR_DATA_ROOT="/Volumes/Love SSD"
AUTOHDR_OUTPUT_ROOT="/Volumes/Love SSD/AutoHDR_Submissions"
AUTOHDR_CHECKPOINT_ROOT="/Volumes/Love SSD/AutoHDR_Checkpoints"
KAGGLE_API_TOKEN=<token>
KAGGLE_MCP_URL=https://www.kaggle.com/mcp
KAGGLE_USERNAME=<optional, for Kaggle CLI>
KAGGLE_KEY=<optional, for Kaggle CLI>
AKASH_API_KEY=<optional>
AKASH_TRAIN_SCRIPT_URL=<optional, only for Akash deploy>
AUTOHDR_BOUNTY_TEAM_NAME=<optional>
AUTOHDR_BOUNTY_EMAIL=<optional>
AUTOHDR_BOUNTY_KAGGLE_USERNAME=<optional>
AUTOHDR_BOUNTY_GITHUB_REPO=<optional>
```

Defaults are loaded from `backend/config.py` and `.env`.

---

## 4. Scoring Reality

We do not compute the final hidden competition score locally.  
Operationally, bounty scoring should be treated as **external and composite**; use real scored submissions as the final arbiter.

Practical rule:

- Use MAE/SSIM/PSNR for local diagnostics only.
- Promote methods based on real score deltas, not proxy wins alone.

---

## 5. Canonical Commands

Run all commands from repo root: `/Users/aaron/Desktop/AutoHDR`.

### 5.1 Heuristic baseline and dimension-bucket micro-grid

```bash
python -m backend.scripts.heuristics.heuristic_baseline --phase both
python -m backend.scripts.heuristics.heuristic_dim_bucket --search --search-pairs 500
python -m backend.scripts.heuristics.heuristic_dim_bucket
```

Expected artifacts: `corrected_dim_bucket/` and `submission_dim_bucket_microgrid.zip`.

### 5.1.1 Kaggle dataset download (CLI helper)

```bash
AUTOHDR_DATA_ROOT="/Volumes/Love SSD" ./download_kaggle.sh
```

### 5.1.2 CalibGuard-Dim (v5 successor) workflow

```bash
python -m backend.scripts.heuristics.build_calibguard_dim_table \
  --train-dir "/Volumes/Love SSD/lens-correction-train-cleaned" \
  --sample-per-dim 300 \
  --min-dim-support 30 \
  --out-table "/Users/aaron/Desktop/AutoHDR/backend/scripts/heuristics/calibguard_dim_table.json" \
  --out-manifest "/Users/aaron/Desktop/AutoHDR/backend/scripts/heuristics/calibguard_dim_manifest.json"

python -m backend.scripts.heuristics.heuristic_calibguard_dim \
  --test-dir "/Volumes/Love SSD/test-originals" \
  --table "/Users/aaron/Desktop/AutoHDR/backend/scripts/heuristics/calibguard_dim_table.json" \
  --profile safe \
  --output-dir "/Volumes/Love SSD/AutoHDR_Submissions" \
  --artifact-tag calibguard_dim

python -m backend.scripts.heuristics.heuristic_calibguard_dim \
  --test-dir "/Volumes/Love SSD/test-originals" \
  --table "/Users/aaron/Desktop/AutoHDR/backend/scripts/heuristics/calibguard_dim_table.json" \
  --profile balanced \
  --output-dir "/Volumes/Love SSD/AutoHDR_Submissions" \
  --artifact-tag calibguard_dim

python -m backend.scripts.heuristics.heuristic_calibguard_dim \
  --test-dir "/Volumes/Love SSD/test-originals" \
  --table "/Users/aaron/Desktop/AutoHDR/backend/scripts/heuristics/calibguard_dim_table.json" \
  --profile aggressive \
  --output-dir "/Volumes/Love SSD/AutoHDR_Submissions" \
  --artifact-tag calibguard_dim

python -m backend.scripts.heuristics.analyze_scored_submission \
  --score-csv "/Users/aaron/Desktop/AutoHDR/submission.csv" \
  --test-dir "/Volumes/Love SSD/test-originals" \
  --out-json "/Users/aaron/Desktop/AutoHDR/backend/scripts/heuristics/calibguard_scored_analysis.json"
```

Optional hot-path flags (additive, no command replacement):

```bash
# Table build with quality controls (opt-in)
python -m backend.scripts.heuristics.build_calibguard_dim_table \
  --interp lanczos4 \
  --border-mode reflect \
  --metric-profile competition_lite \
  --alpha-grid 0.0,0.1 \
  --optimizer grid

# Optional SciPy global search (bounded + timeout guarded)
python -m backend.scripts.heuristics.build_calibguard_dim_table \
  --optimizer de \
  --alpha-grid 0.0,0.1,0.2

# Inference render override (otherwise table/default settings are used)
python -m backend.scripts.heuristics.heuristic_calibguard_dim \
  --profile safe \
  --interp lanczos4 \
  --border-mode reflect
```

### 5.1.3 Submission artifact hygiene (list + prune)

```bash
python -m backend.scripts.heuristics.prune_submission_artifacts --list

python -m backend.scripts.heuristics.prune_submission_artifacts \
  --zip-name submission_dim_bucket_microgrid_safezero_v1.zip \
  --zip-name submission_dim_bucket_microgrid_zero_guard_v1_20260222_030358.zip \
  --zip-name submission_dim_bucket_microgrid_zero_guard_v2_20260222_030614.zip \
  --zip-name submission_dim_bucket_microgrid_zero_guard_v3_20260222_030901.zip

python -m backend.scripts.heuristics.prune_submission_artifacts \
  --zip-name submission_dim_bucket_microgrid_safezero_v1.zip \
  --zip-name submission_dim_bucket_microgrid_zero_guard_v1_20260222_030358.zip \
  --zip-name submission_dim_bucket_microgrid_zero_guard_v2_20260222_030614.zip \
  --zip-name submission_dim_bucket_microgrid_zero_guard_v3_20260222_030901.zip \
  --apply
```

### 5.1.4 Bounty submission automation (zip -> scored CSV)

```bash
python -m backend.scripts.bounty_submit \
  --zip-file "/Volumes/Love SSD/AutoHDR_Submissions/submission_calibguard_cycle1_safe_20260222_102221.zip" \
  --out-csv "/Users/aaron/Desktop/AutoHDR/backend/outputs/bounty/scored_calibguard_cycle1_safe.csv" \
  --out-summary-json "/Users/aaron/Desktop/AutoHDR/backend/outputs/bounty/summary_calibguard_cycle1_safe.json"
```

Notes:

1. `backend/scripts/bounty_submit.py` now loads repo `.env` automatically (`AUTOHDR_BOUNTY_TEAM_NAME`, `AUTOHDR_BOUNTY_EMAIL`, `AUTOHDR_BOUNTY_KAGGLE_USERNAME`, `AUTOHDR_BOUNTY_GITHUB_REPO`).
2. Do not run concurrent submissions for the same zip; it can create duplicate bounty jobs.
3. Optional CLI overrides still work: `--team-name`, `--email`, `--kaggle-username`, `--github-repo`.

### 5.2 Local training / inference / evaluation

```bash
python -m backend.scripts.local_training.train --epochs 1 --max-train 32 --max-val 8 --batch-size 2
python -m backend.scripts.local_training.inference --checkpoint /Volumes/Love\ SSD/AutoHDR_Checkpoints/best_model.pt --limit 25
python -m backend.scripts.local_training.evaluate_model --checkpoint /Volumes/Love\ SSD/AutoHDR_Checkpoints/best_model.pt --max-val 64
```

### 5.3 Kaggle notebook execution (canonical)

```bash
python -m backend.scripts.kaggle.run_notebook_mcp status --owner alwoods --slug train-unet-lens-correction
python -m backend.scripts.kaggle.run_notebook_mcp run --owner alwoods --slug train-unet-lens-correction --enable-gpu
python -m backend.scripts.kaggle.run_notebook_mcp run-and-wait --owner alwoods --slug train-unet-lens-correction --enable-gpu --wait-timeout-min 120
```

### 5.3.1 Kaggle hybrid notebook (parallel experiment)

Notebook file:

- `/Users/aaron/Desktop/AutoHDR/backend/scripts/kaggle/Train-Hybrid-CalibGuard.ipynb`

Operational contract for this notebook:

1. Always emits two zips:
   - `submission_safe_heuristic_<YYYYMMDD_HHMMSS>.zip`
   - `submission_hybrid_candidate_<YYYYMMDD_HHMMSS>.zip`
2. Also emits:
   - `run_summary_<YYYYMMDD_HHMMSS>.json` with `recommended_submission_zip`
   - optional `hybrid_refiner_<YYYYMMDD_HHMMSS>.pt`
3. Promotion remains governed by Section 8 decision gates; do not bypass gates because hybrid mode was selected internally.
4. Treat this as an experiment track until it clears hard blockers on scored outputs.

### 5.4 Akash deployment (secondary path)

```bash
npm run deploy
```

Akash flow now requires `AKASH_TRAIN_SCRIPT_URL` (public URL to a standalone training script) and is secondary to the Kaggle path.

---

## 6. Submission History

| Submission | Approach | Mean Score | Status |
|---|---|---:|---|
| `submission_global_grid.csv` *(legacy: `submission_heuristic.csv`)* | Global baseline (`k1=-0.17`, `k2=0.35`, `alpha=0`) | 24.00 | Baseline |
| `submission_hough_proxy.csv` *(legacy: `submission_v2.csv`)* | Per-image candidate with Hough proxy | 3.45 | Rejected |
| `submission_laplacian_proxy.csv` *(legacy: `submission_v3.csv`)* | Per-image candidate with Laplacian proxy | 21.21 | Rejected |
| `submission_nelder_per_image.csv` *(legacy: `submission_phase2.csv`)* | Nelder-Mead per-image optimization | 18.89 | Rejected |
| `submission_dim_bucket_microgrid.csv` *(legacy: `submission_v4.csv`)* | Dimension-aware bucketed coefficients | **29.64** | Current best |
| `submission_dim_bucket_microgrid_tuned_20260222.csv` *(source export: `submission.csv`)* | 7-bucket micro-grid tuned coefficients | 25.00 | Regression (do not promote) |
| `submission_dim_bucket_microgrid_safezero_v1.csv` | Zero-guard identity on `heavy_crop` + `portrait_cropped` | 24.98 | Regression (hold) |
| `submission_dim_bucket_microgrid_zero_guard_v1_20260222_030358.csv` | Same policy as `safezero_v1` with timestamped artifact | 24.98 | Regression (duplicate behavior) |
| `submission_dim_bucket_microgrid_zero_guard_v2_20260222_030614.csv` | Added `portrait_standard` identity guard | 24.91 | Regression (reject) |
| `submission_dim_bucket_microgrid_zero_guard_v3_20260222_030901.csv` | Looser `moderate_crop` coefficients (`-0.06`, `0.12`) | 24.69 | Regression (reject) |
| `submission_calibguard_cycle1_safe_scored.csv` | CalibGuard cycle-1 safe profile (`submission_calibguard_cycle1_safe_20260222_102221.zip`) scored via bounty API | 18.17 | Regression (reject) |
| `v11_submission.csv` | Kaggle v11 `best_model.pt` -> local inference zip (`kaggle_v11_submission.zip`) | **0.00** | Hard reject (all rows zero) |
| `submission_calibguard_dim_<profile>_<timestamp>.csv` | CalibGuard-Dim exact-dimension guardrail routing | Pending | Active iteration |

### 6.1 Tested vs Untested (Readable Board)

**Tested and scored (12 total):**

1. `submission_dim_bucket_microgrid.csv` — 29.64 (best)
2. `submission_dim_bucket_microgrid_tuned_20260222.csv` — 25.00 (regression)
3. `submission_dim_bucket_microgrid_safezero_v1.csv` — 24.98 (regression)
4. `submission_dim_bucket_microgrid_zero_guard_v1_20260222_030358.csv` — 24.98 (duplicate behavior)
5. `submission_dim_bucket_microgrid_zero_guard_v2_20260222_030614.csv` — 24.91 (regression)
6. `submission_dim_bucket_microgrid_zero_guard_v3_20260222_030901.csv` — 24.69 (regression)
7. `submission_global_grid.csv` — 24.00 (baseline)
8. `submission_laplacian_proxy.csv` — 21.21 (rejected)
9. `submission_nelder_per_image.csv` — 18.89 (rejected)
10. `submission_calibguard_cycle1_safe_scored.csv` — 18.17 (regression)
11. `submission_hough_proxy.csv` — 3.45 (rejected)
12. `v11_submission.csv` — 0.00 (hard reject)

**Untested / pending score:**

1. `submission_calibguard_dim_<profile>_<timestamp>.csv`

**Still running / building now:**

1. `build_calibguard_dim_table` local build process (see Section 1.1 command line).

### 6.2 Naming Convention (Use This Going Forward)

To reduce confusion across agents and runs:

1. Use descriptive canonical names in history, never `vN` alone.
2. Submission artifact naming pattern:
`submission_<method_family>_<variant_or_profile>_<YYYYMMDD_HHMMSS>.zip`
3. Scored CSV naming pattern (local bookkeeping):
`scored_<method_family>_<variant_or_profile>_<YYYYMMDD_HHMMSS>.csv`
4. If you use a short alias in chat, always include canonical mapping once in the same message:
`alias -> canonical_name`.

Example:

- `calib-safe-a -> submission_calibguard_dim_safe_20260222_141500.zip`

---

## 7. Regression Forensics — February 22, 2026

### 7.1 Headline metrics (from scored export)

Source: `/Users/aaron/Desktop/AutoHDR/submission.csv` (1000 rows)

- Mean score: **24.9984**
- Median score: **26.555**
- Zero-score rate: **15.9%** (**159/1000**)
- Delta vs current best (`29.64`): **-4.64**

### 7.2 Bucket-level failure profile

| Bucket | Count | Mean | Zero-rate |
|---|---:|---:|---:|
| `standard` | 467 | 30.343 | 8.35% |
| `near_standard_short` | 341 | 22.640 | 18.18% |
| `near_standard_tall` | 94 | 23.890 | 14.89% |
| `moderate_crop` | 68 | 10.879 | 27.94% |
| `heavy_crop` | 21 | 0.633 | 85.71% |
| `portrait_cropped` | 6 | 5.620 | 83.33% |
| `portrait_standard` | 3 | 25.110 | 66.67% |

### 7.3 Proxy mismatch finding

- On 1,500 sampled training pairs, tuned coefficients improved local proxy MAE:
  - old MAE: **4.9738**
  - tuned MAE: **4.8785**
  - delta (tuned-old): **-0.0953**
- Hidden scored result still regressed materially, so local train MAE cannot be used as a sole promotion criterion.

### 7.4 Root cause statement

Micro-grid tuning overfit MAE-proxy behavior and increased hidden-score failure modes in crop-sensitive buckets.

### 7.5 Process contract changes (no runtime API/type changes)

- Artifact naming must be immutable and timestamped: `<method>_<YYYYMMDD_HHMMSS>.zip`.
- Scored result CSV must be mapped to a run manifest before any promotion decision.
- Best-scored artifact paths are immutable once established.

### 7.6 Validation scenarios

1. Recompute mean/median/zero-rate from `/Users/aaron/Desktop/AutoHDR/submission.csv`; confirm **24.9984**, **26.555**, **15.9%**.
2. Recompute bucket metrics by joining `/Users/aaron/Desktop/AutoHDR/submission.csv` with `/Volumes/Love SSD/test-originals/*.jpg` dimensions and current bucket classifier.
3. Validate this runbook contains:
   - regression row in submission history,
   - forensics section with metrics and root-cause note,
   - hard blocker thresholds in decision gates,
   - provenance controls in reliability checklist.
4. Enforce that candidates failing any hard blocker are never promoted, even if proxy metrics improve.

### 7.7 Assumptions

- `/Users/aaron/Desktop/AutoHDR/submission.csv` corresponds to the latest tuned micro-grid submission.
- `29.64` remains the current best reference score until a gated candidate exceeds it.
- Regression label default: `submission_dim_bucket_microgrid_tuned_20260222.csv`.

### 7.8 Zero-Guard Batch Forensics — February 22, 2026 (~3:33 AM CST)

Downloaded scored exports in repo root:

- `/Users/aaron/Desktop/AutoHDR/tab1-submission.csv`
- `/Users/aaron/Desktop/AutoHDR/tab2-submission.csv`
- `/Users/aaron/Desktop/AutoHDR/tab3-submission.csv`
- `/Users/aaron/Desktop/AutoHDR/tab4-submission.csv`

Deterministic mapping from score-diff fingerprints:

- `tab2-submission.csv` maps to `zero_guard_v2` (only `portrait_standard` rows changed; 2 scored rows changed).
- `tab3-submission.csv` maps to `zero_guard_v3` (`moderate_crop` rows changed; 49 scored rows changed).
- `tab1-submission.csv` and `tab4-submission.csv` are identical and map to the pair `{safezero_v1, zero_guard_v1}` in unknown order (both corrected folders are byte-identical across 1000 images).

Headline metrics:

| Scored CSV | Mean | Median | Zero-rate |
|---|---:|---:|---:|
| `tab1-submission.csv` | 24.9839 | 26.53 | 15.3% |
| `tab2-submission.csv` | 24.9100 | 26.47 | 15.2% |
| `tab3-submission.csv` | 24.6893 | 26.30 | 15.9% |
| `tab4-submission.csv` | 24.9839 | 26.53 | 15.3% |

Key bucket observations:

- `zero_guard_v2` reduced `portrait_standard` zeros but collapsed `portrait_standard` mean from `25.110` to `0.457` (hard reject signal).
- `zero_guard_v3` materially harmed `moderate_crop`: mean `10.879` -> `6.546`, zero-rate `27.94%` -> `36.77%` (hard reject signal).
- `safezero_v1`/`zero_guard_v1` improved `heavy_crop` vs tuned baseline (`mean 0.633` -> `1.315`, zero-rate `85.71%` -> `66.67%`) but remained globally regressive.

Decision from this batch:

- Do not promote any of the four candidates.
- Keep only the duplicated `safezero_v1`/`zero_guard_v1` result as a guardrail reference point.

### 7.9 Qualitative Synthesis — Why We Have Not Exceeded 30

Across recent iterations, the performance ceiling is not from one bug; it is a repeated pattern of objective mismatch and robustness debt.

1. Proxy optimization drift:
Local MAE improvements did not translate to hidden score improvements; edge/structure-heavy hidden scoring penalizes geometric mistakes more than MAE captures.
2. Persistent hard-fail pressure:
Recent heuristic runs stay in the ~15% zero-rate band above promotion threshold (`<=12%`); a small set of catastrophic failures has outsized impact on mean score.
3. Crop-sensitive bucket collapse:
`moderate_crop`, `heavy_crop`, and `portrait_cropped` remain the dominant drag buckets; guardrail variants reduce one failure mode while creating another in adjacent buckets.
4. Coefficient granularity is still too coarse:
Bucket-level `k1/k2` tuning is not enough for heterogeneous lens/dimension regimes; exact-dimension behavior is partially captured but not yet fully promoted with robust routing.
5. Iteration efficiency leakage:
Near-duplicate submissions consumed cycles without producing new information; promotion decisions were occasionally delayed by weak provenance mapping between artifacts and scored CSVs.
6. Model track immaturity:
v11 scored `0.0000` with `100%` zeros, showing structural failure against hidden hard-fail rules; current model outputs are not yet a safe promotion path.

### 7.10 Implications For Next Iterations

1. Treat zero-rate reduction and score uplift as co-primary objectives; reject single-metric wins.
2. Require per-bucket deltas for every scored candidate before promotion decisions.
3. Prioritize crop-sensitive robustness over broad global retuning.
4. Enforce novelty check before submission; do not re-submit byte-identical behavior.
5. Keep model track in diagnostic mode until it clears hard-fail gates on held-out checks.

These implications are enforceable via Decision Gates in Section 8.

---

## 8. Decision Gates

Use these gates when choosing final submission strategy:

1. **Hard blockers (must pass before any promotion):**
   - reject any candidate with overall zero-score rate **> 12%**
   - reject any candidate where a bucket with **>= 20 samples** has mean **< 15**
   - reject any candidate where a bucket with **>= 20 samples** has zero-score rate **> 30%**
2. **Evidence completeness gate:**
   - reject any promotion decision that lacks per-bucket scored deltas vs current best and vs immediate baseline
   - reject any promotion decision without artifact-to-scored-CSV provenance mapping
3. **Proxy mismatch gate:**
   - reject promotion if improvement evidence is proxy-only (MAE/SSIM/PSNR) without scored uplift
   - treat proxy wins as diagnostic only unless hidden score improves
4. **Novelty gate (submission efficiency):**
   - reject re-submission of byte-identical output behavior under new naming
   - verify candidate differs from latest baseline before upload
   - record canonical run name in Submission History; do not use `vN` shorthand alone
5. **Heuristic promotion gate:**
   - keep/advance heuristic path only if mean score is at least current best minus 1.0 point
   - zero-rate and crop-sensitive bucket metrics must be non-regressive
6. **Model promotion gate (diagnostic until proven stable):**
   - promote model path only if overall zero-rate is **<= 12%** and no hard blocker fails
   - scored mean must beat `submission_dim_bucket_microgrid.csv` by at least 1.5 points
   - output quality must remain stable across orientation/crop buckets
7. **Fallback rule near deadline:**
   - if model evidence is incomplete by final 60 minutes, submit best verified heuristic artifact.

---

## 9. Reliability Checklist

Before producing or uploading a submission:

1. Validate config resolution:
   - `AUTOHDR_DATA_ROOT` exists
   - `test-originals/` and `lens-correction-train-cleaned/` are present
2. Confirm script entrypoint works:
   - `python -m ... --help` exits 0
3. Confirm outputs are generated under configured roots:
   - checkpoints in `AUTOHDR_CHECKPOINT_ROOT`
   - zips and corrected images in `AUTOHDR_OUTPUT_ROOT`
4. Log artifact paths and command used for each run.
5. Enforce immutable artifact naming:
   - format: `<method>_<YYYYMMDD_HHMMSS>.zip`
6. Never overwrite a path that belongs to a best-scored artifact.
7. Log a per-run manifest containing:
   - git commit hash
   - exact command
   - coefficient set
   - output artifact path
   - scored CSV path

---

## 10. Known Constraints

- Local Apple Silicon can be unstable for larger model runs.
- External SSD I/O can bottleneck listing/scan-heavy operations.
- Hidden scoring behavior can diverge from local proxy metrics.
- Kaggle execution depends on valid MCP token and notebook permissions.

---

## 11. File Map

```text
/Users/aaron/Desktop/AutoHDR/
├── backend/
│   ├── config.py
│   ├── core/
│   ├── evaluation/
│   └── scripts/
│       ├── akash/akash_deploy.ts
│       ├── bounty_submit.py
│       ├── heuristics/
│       │   ├── heuristic_baseline.py
│       │   ├── heuristic_dim_bucket.py (canonical implementation)
│       │   ├── build_calibguard_dim_table.py
│       │   ├── heuristic_calibguard_dim.py
│       │   ├── analyze_scored_submission.py
│       │   ├── calibguard_dim_table.json
│       │   ├── calibguard_dim_manifest.json
│       │   ├── analyze_dimensions.py
│       │   ├── diagnose_dim_bucket_errors.py (canonical implementation)
│       │   └── prune_submission_artifacts.py
│       ├── kaggle/
│       │   ├── run_notebook_mcp.py
│       │   ├── Train-UNet-Lens-Correction.ipynb
│       │   └── Train-Hybrid-CalibGuard.ipynb (parallel experiment)
│       └── local_training/
│           ├── train.py
│           ├── inference.py
│           └── evaluate_model.py
├── demo.py
├── docs/ops/
│   ├── research_log.md (deprecated pointer to this runbook)
│   └── agent_notes.md (deprecated pointer to this runbook)
├── .env.example
└── requirements.txt
```

---

## 12. Execution Log — February 22, 2026 (Kaggle + Local)

### 12.1 Authentication and connectivity work completed

1. Kaggle credentials were validated locally from repo `.env` and `~/.kaggle/access_token`.
2. Kaggle MCP connectivity was confirmed with:
   - `python -m backend.scripts.kaggle.run_notebook_mcp status --owner alwoods --slug train-unet-lens-correction`

### 12.2 Kaggle MCP runner changes completed

File updated: `/Users/aaron/Desktop/AutoHDR/backend/scripts/kaggle/run_notebook_mcp.py`

1. Added `--local-notebook-path` upload path to push local notebook source before `save_notebook`.
2. Fixed `run-and-wait` logic to wait for terminal state instead of returning early on `RUNNING`.
3. Added `session-output` command to retrieve/tail notebook session logs.
4. Added polling controls (`--poll-interval-sec`) and log tail controls (`--log-tail-lines`).
5. Added graceful handling when session-output endpoint returns HTTP 400 for terminal sessions.

### 12.3 Kaggle notebook (`Train-UNet-Lens-Correction.ipynb`) changes completed

File updated: `/Users/aaron/Desktop/AutoHDR/backend/scripts/kaggle/Train-UNet-Lens-Correction.ipynb`

1. Removed embedded hardcoded token.
2. Added robust dataset discovery under `/kaggle/input` and extracted roots.
3. Added fallback download path when mounted inputs are missing.
4. Iterated fallback implementation to resolve hard failures:
   - v9 failure: `No space left on device` while attempting full data handling.
   - v10 failure: Kaggle API response shape mismatch (`ApiListDataFilesResponse` non-iterable).
5. Final fallback behavior:
   - Supports bounded training-pair download (`--fallback-max-pairs`, default 3000).
   - Handles Kaggle list-files response variants (`.files`, list, iterable).
6. Restored clean notebook execution entrypoint (`if __name__ == '__main__': main()`).

### 12.4 Kaggle run timeline and outcomes

Notebook: `alwoods/train-unet-lens-correction`

1. v8: `RUNNING` with unstable output visibility; superseded by later fixes.
2. v9: `ERROR` due to disk exhaustion (`No space left on device`).
3. v10: `ERROR` due to fallback API response handling bug.
4. v11: `COMPLETE` at ~09:40 UTC with training artifacts present.

Observed v11 output artifacts (Kaggle session files):

- `best_model.pt`
- `training_history.json`
- `checkpoint_epoch*.pt`

### 12.5 Local artifact pulls and inference work

1. Downloaded v11 artifacts to:
   - `/Volumes/Love SSD/AutoHDR_Checkpoints/kaggle/train-unet-lens-correction/v11/best_model.pt`
   - `/Volumes/Love SSD/AutoHDR_Checkpoints/kaggle/train-unet-lens-correction/v11/training_history.json`
2. Ran local inference from v11 checkpoint to build submission zip:
   - output zip: `/Volumes/Love SSD/AutoHDR_Submissions/inference/kaggle_v11_submission/kaggle_v11_submission.zip`
   - output summary: `/Volumes/Love SSD/AutoHDR_Submissions/inference/kaggle_v11_submission/summary.json`

### 12.6 Scored result and postmortem

Input scored CSV: `/Users/aaron/Desktop/AutoHDR/v11_submission.csv`

1. Result: mean **0.0000**, median **0.0000**, zero-score rate **100% (1000/1000)**.
2. Decision: hard reject model output; do not promote any v11-derived submission artifact.
3. Side-analysis on `/Volumes/Love SSD`:
   - `kaggle_v11_submission.zip` is much smaller (~172 MB) than heuristic zips (~568-570 MB).
   - Zip integrity is valid (1000 unique files, dimensions preserved).
   - Compression delta traced to over-smoothed/low-detail outputs (severe texture collapse).

### 12.7 Heuristic and operational support work completed

1. Updated heuristic execution script:
   - `/Users/aaron/Desktop/AutoHDR/backend/scripts/heuristics/heuristic_dim_bucket.py`
   - Added safe-fallback controls, dimension-coefficient search/load paths, and run-source tracking.
2. Generated submission artifacts for zero-guard variants under:
   - `/Volumes/Love SSD/AutoHDR_Submissions/`
3. Added and maintained operator notes file:
   - `/Users/aaron/Desktop/AutoHDR/docs/ops/agent_notes.md`

### 12.8 Bounty automation validation and findings

1. Verified bounty submit flow from site bundle and API:
   - `POST /api/upload-url` -> presigned S3 upload
   - `PUT <upload_url>` for zip upload
   - `POST /api/score` + polling `GET /api/score?request_id=...`
2. Reused existing script:
   - `/Users/aaron/Desktop/AutoHDR/backend/scripts/bounty_submit.py`
   - Patch applied: script now auto-loads repo `.env` using `python-dotenv`.
3. Configured identity env vars in local `.env`:
   - `AUTOHDR_BOUNTY_TEAM_NAME="Aaron"`
   - `AUTOHDR_BOUNTY_EMAIL="aaronlorenzowoods@gmail.com"`
   - `AUTOHDR_BOUNTY_KAGGLE_USERNAME="alwoods"`
4. Live run executed on:
   - zip: `/Volumes/Love SSD/AutoHDR_Submissions/submission_calibguard_cycle1_safe_20260222_102221.zip`
   - request id: `1031b2da-202c-465a-9272-47fdf9fd86a7`
   - completion timestamp (UTC): `2026-02-22T10:32:28.966304`
   - elapsed: `216.6s`
5. Output artifacts written to:
   - `/Users/aaron/Desktop/AutoHDR/backend/outputs/bounty/submission_calibguard_cycle1_safe_scored.csv`
   - `/Users/aaron/Desktop/AutoHDR/backend/outputs/bounty/submission_calibguard_cycle1_safe_summary.json`
6. Scored outcome:
   - CSV mean score: `18.1704` (scale 0-100)
   - CSV median score: `18.095`
   - zero-score rate: `16.6%` (`166/1000`)
   - summary hard fails: `9`
   - missing images: `0`
   - summary `avg_score`: `0.1817` (normalized scale; equals `18.17/100`)
7. Decision:
   - hard reject for promotion (material regression vs current best `29.64`).
8. Duplicate-submission finding:
   - Two concurrent `bounty_submit.py` processes were observed uploading the same zip in parallel.
   - One duplicate process was terminated to prevent duplicate scoring jobs.
   - New operational rule: one active bounty submit process per artifact.

### 12.9 Linting and cleanup pass learnings (February 22, 2026)

1. Python cleanup:
   - Ran `ruff check --fix` across tracked Python files and then `ruff format`.
   - Result: consistent import ordering and formatting across backend scripts/tests and helper tooling.
2. Markdown linting:
   - Ran `markdownlint-cli` across tracked Markdown files.
   - Legacy docs (`plan.md` and archived notes) contain intentional long operational lines, so strict `MD013` is noisy for this repo.
   - Practical policy: lint Markdown with `MD013` and `MD041` disabled unless a future doc style guide explicitly enforces hard wrapping.
3. Repo hygiene lesson:
   - Parallel worktrees/copies created many duplicate numbered files in prior runs.
   - `git clean -fd` after branch checkpointing is an effective way to recover a clean workspace safely.
