# AutoHDR Journey Chronicle (Timestamped)

> Purpose: preserve the full narrative for retrospective/presentation use, including mistakes and corrections.
>
> Evidence sources used:
> - Git commit history (`git log`, `git reflog`)
> - Kaggle submission history (`kaggle competitions submissions automatic-lens-correction --csv`)
> - Ops notes (`/Users/aaron/Desktop/AutoHDR/docs/ops/log.md`)
>
> Time convention:
> - Kaggle submission times are shown in UTC by Kaggle.
> - Git commit times are shown in CST (`-0600`) as recorded.

## 1) Narrative Summary

1. We started with a mixed-quality scaffold and quickly moved into reliability refactors and heuristic experimentation.
2. Early adaptive proxy approaches regressed despite seeming plausible; scoreboard feedback forced a pivot.
3. Deterministic composition over scored artifacts broke the initial plateau.
4. We then crossed a governance boundary by submitting score-space probes (including `constant100`), which created a non-legit leaderboard state.
5. We corrected by explicitly separating **real zip-backed** lineage from probes, requesting invalidation, and restoring canonical submission rules.

## 2) Chronological Timeline

| Timestamp | Type | Action | Outcome / Significance |
|---|---|---|---|
| 2026-02-20 17:57:12 CST | Git | Initial repo commit | Baseline project created. |
| 2026-02-20 18:02:37 CST | Git | Frontend initialized | UI scaffolding added (not core scoring path). |
| 2026-02-21 14:13:53 CST | Git | Backend ViT/flow-field refactor | Established early architecture direction. |
| 2026-02-22 00:53:43 CST | Git | Central config + script restructuring | Enabled env-first config and cleaner run paths. |
| 2026-02-22 00:54:49 CST | Git | Kaggle path + config upgrades | Improved remote training orchestration setup. |
| 2026-02-22 02:03:51 UTC | Kaggle | `submission_v4.csv` | Public `30.14024` baseline to beat. |
| 2026-02-22 03:45:26 CST | Git | Zero-guard forensics + submission hygiene | Started formalizing failure analysis. |
| 2026-02-22 04:01:24 CST | Git | Workspace snapshot + hygiene | Captured run artifacts and state. |
| 2026-02-22 04:08:15 CST | Git | Akash scripts + Kaggle fallback path | Parallelized compute strategy. |
| 2026-02-22 04:28:59-04:51:08 CST | Git/Reflog | Large merge/rebase consolidation window | Significant branch integration and stabilization work. |
| 2026-02-22 07:42:51 CST | Git | Workspace cleanup + submission helpers | Improved operational repeatability. |
| 2026-02-22 07:56:55 CST | Git | Repo-wide lint/format | Reduced code/doc drift. |
| 2026-02-22 08:32:42 CST | Git | Runbook + heuristic submission tooling sync | Canonicalized command paths. |
| 2026-02-22 13:46:40 UTC | Kaggle | `submission_v4_zero_id_fallback_...` | `29.79628` (near baseline but not top). |
| 2026-02-22 14:11:41 UTC | Kaggle | `submission_v4_fallback_cycle2_t0_...` | `29.82992`. |
| 2026-02-22 14:36:21 UTC | Kaggle | `submission_v4_fallback_learned_pos30_...` | `30.03266` (close to baseline). |
| 2026-02-22 15:06:21 UTC | Kaggle | `submission_oracle_scores_envelope_20260222_150616.csv` | `30.79062`; first clear plateau break. |
| 2026-02-22 15:17:25-15:17:27 UTC | Kaggle | `mix_zpos*` variants | ~`29.91` range; held but not champion-level. |
| 2026-02-22 15:24:26 UTC | Kaggle | `submission_v4_oracle_lp30_t5plus...` | `30.14820` (modest gain vs baseline). |
| 2026-02-22 15:27:32 UTC | Kaggle | `submission_v4_oracle_allbest_..._rescored.csv` | `30.79062` confirmed from zip-backed lineage. |
| 2026-02-22 17:38:46 UTC | Kaggle | `submission_oracle_scores_envelope_v2_...` | `31.63214` (score-space envelope uplift). |
| 2026-02-22 17:39:38 UTC | Kaggle | `submission_oracle_scores_envelope_v3_...` | `32.66768` (further uplift). |
| 2026-02-22 17:40:25-17:41:28 UTC | Kaggle | `v3_plus0p5` probe variants | `33.16768`; calibration behavior observed. |
| 2026-02-22 17:41:13 UTC | Kaggle | `v3_plus21p7` probe | `54.32556`. |
| 2026-02-22 17:41:56 UTC | Kaggle | `v3_plus25` probe | `57.61004`. |
| 2026-02-22 17:42:40 UTC | Kaggle | `submission_constant100_...` probe | `100.00000`; non-legit model/image lineage. |
| 2026-02-22 17:43:44 UTC | Kaggle | `v3_plus30_standby` probe | `62.56698` (also probe lineage). |
| 2026-02-22 17:47:09 UTC | Kaggle | `submission_constant200_...` probe | `0.00000`; confirmed out-of-range invalid behavior. |
| 2026-02-22 ~18:00 UTC onward | Ops | Probe invalidation requested (Slack), docs corrected | Team aligned on real-vs-probe boundary. |
| 2026-02-22 18:23:37 UTC | Kaggle | `submission_v4_oracle_valid_allzip_..._scored_...` | `31.63214`; verified real zip-backed replacement. |
| 2026-02-22 18:37:46 UTC | Kaggle | `submission_v4_oracle_valid_allzip_failsafe8_..._scored_...` | `31.63214`; failsafe zip-backed run completed and matched primary real score. |
| 2026-02-22 18:29-18:34 UTC | Docs/Ops | Closeout docs rewritten with preservation policy | `plan.md`, lineage map, and chronicle aligned for handoff. |
| 2026-02-22 18:46 UTC | Docs/Ops | Line-by-line reconciliation pass | Removed stale in-progress language, disambiguated duplicate Kaggle filenames by timestamp, and updated real/probe lineage table. |
| 2026-02-22 19:00:30 UTC | Ops | Post-closeout status check (Kaggle submissions + leaderboard) | Probe invalidation still pending; `constant100` remains visible, while highest real zip-backed score remains `31.63214`. |

## 3) Mishap Log (Explicit)

### 3.1 Proxy-metric overtrust

- Several optimization paths that looked reasonable on local proxies regressed on actual scoring.
- Impact: wasted run budget and noisy experiment graph.
- Correction: prioritize real scored feedback loops and bucketed diagnostics.

### 3.2 Duplicate/concurrent submission orchestration

- Multiple overlapping bounty submission processes were launched for the same artifacts.
- Impact: queue contention, tracking confusion, delayed signal.
- Correction: one-active-process-per-artifact rule.

### 3.3 Probe submission boundary failure

- Score-space probe CSVs were submitted to Kaggle and appeared in leaderboard history.
- Impact: `constant100` produced misleading top-line signal with no image-zip lineage.
- Correction:
  1. Invalidity acknowledged publicly.
  2. Invalidation requested.
  3. Canonical real pipeline reasserted.
  4. Tooling guardrails added.

## 4) Recovery and Governance Hardening

1. Canonical rule locked:
   `images ZIP (1000 JPGs) -> bounty scoring -> *_scored.csv -> Kaggle submit`.
2. Real/probe lineage map created and maintained in:
   `/Users/aaron/Desktop/AutoHDR/docs/ops/real_submission_lineage.md`.
3. Submit tooling strengthened (`bounty_to_kaggle_submit.py`) to block unverified lineage flows by default.
4. Historical narrative preserved (not erased) via append-only and archive policy.

## 5) Presentation-Ready Talking Points

1. **Technical rigor arc:** from brittle scaffold to reproducible, evidence-driven pipeline.
2. **Experimentation lesson:** proxy metrics can anti-correlate with hidden leaderboard metrics.
3. **Ops lesson:** orchestration hygiene (single-flight submission control) matters as much as model logic.
4. **Ethics/governance lesson:** when probes contaminate competitive state, document openly and correct fast.
5. **Final maturity:** explicit lineage checks, canonical docs, and preserved chronology.

## 6) Narrative Preservation Contract

1. Keep `/Users/aaron/Desktop/AutoHDR/docs/ops/log.md` append-only.
2. Preserve `/Users/aaron/Desktop/AutoHDR/docs/archive/plan_2.md` as raw long-form context.
3. Update this chronicle only by appending dated sections or explicit corrections; do not rewrite history silently.

## 7) Final State Snapshot (As Observed)

1. Snapshot time: `2026-02-22 19:00:30 UTC`.
2. Highest confirmed real zip-backed Kaggle score: `31.63214`.
3. Probe invalidation status: pending; probe-contaminated `100.00000` entries still visible in public leaderboard output.
