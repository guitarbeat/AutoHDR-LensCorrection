# Speaker Notes By Slide (Copy/Paste Ready)

## Slide 1 - Problem, Metric, Validity Rule

Primary notes:
"We framed this as a scientific correction problem, not a proxy game. Our validity rule is strict: only ZIP-backed lineage submissions count as real evidence, and our current top real score is 31.63214."

Alt short line:
"Real chain only, real score only: that is our acceptance criterion."

## Slide 2 - Method Science

Primary notes:
"The correction core is Brown-Conrady undistortion using dimension-aware coefficients. We route by geometry buckets because portrait and heavy-crop dimensions behave differently from standard landscape and require safer parameterization."

Alt short line:
"Same distortion model, different geometry buckets, different stability outcomes."

## Slide 3 - Experimental Arc

Primary notes:
"In real lineage, we started at 30.14024 with submission_v4.csv, then reached 30.79062 with oracle_allbest fusion, and finally tied at 31.63214 with valid allzip variants. CalibGuard cycle1 and cycle2 underperformed as final candidates, so they remained diagnostic branches."

Alt short line:
"Baseline, plateau break, top tie, plus clear rejected branches."

## Slide 4 - Top Solution Internals

Primary notes:
"The winning mechanism is deterministic per-image source selection using max_per_image_across_inputs. Source composition was 572, 178, 145, 64, and 41 images across key inputs, then failsafe8 patched 8 risky IDs to harden edge behavior."

Alt short line:
"Per-image fusion gave the lift; failsafe patching kept it stable."

## Slide 5 - Validation and Governance

Primary notes:
"Probe submissions happened and were explicitly excluded from final science claims. We now enforce lineage checks, 1000-row ID parity, and summary consistency. As of 2026-02-22 19:00:30 UTC, invalidation of probe artifacts was still pending."

Alt short line:
"Scientific claims end where lineage verification fails."
