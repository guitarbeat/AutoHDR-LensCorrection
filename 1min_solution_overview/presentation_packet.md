# Presentation Packet (Slides + Script + Notes)

## Slide Plan (Smooth Flow, 5 Slides)

### Slide 1 - Problem And Validity Rule
- **On-slide:** objective, metric context, canonical real chain.
- **Key line:** We optimize correction quality, but only ZIP-backed lineage counts as real evidence.

### Slide 2 - Method Science
- **On-slide:** Brown-Conrady model + dimension-aware routing.
- **Key line:** Geometry-aware coefficient routing stabilized cases where global correction failed.

### Slide 3 - Experiment Arc
- **On-slide:** baseline -> uplift -> top real tie, plus rejected branches.
- **Key line:** Controlled experiments moved us from `30.14024` to `31.63214` in real lineage.

### Slide 4 - Top Solution Internals
- **On-slide:** `max_per_image_across_inputs`, source counts, failsafe8 patch.
- **Key line:** Deterministic per-image fusion gave lift; failsafe patching hardened edge cases.

### Slide 5 - Validation And Governance
- **On-slide:** real vs probe separation, pending invalidation timestamp, guardrails.
- **Key line:** Scientific claims are accepted only when lineage and consistency checks pass.

## 60-Second Script (Timestamped)

Format: `MM:SS-MM:SS -> narration`

`00:00-00:11` -> We treated this as a scientific correction problem: improve lens undistortion quality, but only through real ZIP-backed lineage.

`00:11-00:23` -> The method core is Brown-Conrady undistortion with dimension-aware coefficients, because portrait and heavy-crop geometries behave differently than standard landscape images.

`00:23-00:35` -> In real lineage, baseline `submission_v4.csv` scored `30.14024`, then deterministic fusion in `submission_v4_oracle_allbest...` reached `30.79062`, which was the first durable plateau break.

`00:35-00:49` -> The top real tied score is `31.63214`, validated in two separate ZIP-backed runs, using `max_per_image_across_inputs`: `572`, `178`, `145`, `64`, and `41` images from five scored ZIP-backed sources.

`00:49-01:00` -> Probe submissions happened and were excluded from final science claims. We added strict lineage and `1000`-row/ID guards, plus failsafe8 patching `8` risky IDs. `predicted_mean` `31.19849999999999` stayed an estimate, while real leaderboard evidence remained ZIP-backed.

## Speaker Notes (Copy/Paste)

### Slide 1 Notes
"This is a correction science story, not a proxy-score story. We only treat ZIP-backed lineage submissions as valid evidence."

Alt short line:
"No lineage, no claim."

### Slide 2 Notes
"We use Brown-Conrady undistortion, but route by geometry bucket so risky dimensions do not inherit unstable coefficients from standard landscape cases."

Alt short line:
"Same model, geometry-aware routing."

### Slide 3 Notes
"Real progression was baseline 30.14024, then 30.79062 via oracle allbest fusion, then 31.63214 in the top valid allzip family."

Alt short line:
"Baseline, plateau break, top tie."

### Slide 4 Notes
"Top performance came from deterministic per-image fusion: 572, 178, 145, 64, and 41 images selected from five scored ZIP-backed sources, then failsafe8 patched 8 risky IDs."

Alt short line:
"Per-image fusion plus a targeted patch."

### Slide 5 Notes
"Probe contamination happened, but probes were excluded from final science claims. Guardrails now enforce lineage, 1000-row ID parity, and consistency checks."

Alt short line:
"Scientific validity requires governance validity."
