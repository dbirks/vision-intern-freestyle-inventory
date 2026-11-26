# Experiments

Each YAML specifies a comparison and paths relative to the data mount. Run
`python -m freestyle.experiments.launch <yaml> --dry-run` to inspect the command;
without the flag the launcher creates a unique run directory and starts it. Stage
markers allow `freestyle.experiments.entry` to resume an interrupted run in place.
The launcher refuses to overwrite an existing run. For Kubernetes, stage the data
on the selected node beforehand. Threshold selection uses validation only.

For pseudo-label training set `data.label_source` to `bootstrap` or
`bootstrap+gt` and `data.bootstrap` to a path under `labels/`. Optional
`data.bootstrap_filter` keys are `verdicts` (default `[correct]`),
`min_judge_score`, `min_support`, and `min_id_confidence` (default 0).
Human labels take precedence for images present in both sources. All sources
are filtered by the split map; detector validation holds out training groups.
E1 reports shelf gaps and suggested neighbour votes in per-image results.
Run identity includes tracked changes and non-ignored untracked file contents;
private data and run artifacts are excluded.
