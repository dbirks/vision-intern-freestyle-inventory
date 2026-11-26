# Experiments

Each YAML specifies a comparison and paths relative to the data mount. Run
`python -m freestyle.experiments.launch <yaml> --dry-run` to inspect the command;
without the flag the launcher creates a unique run directory and starts it. Stage
markers allow `freestyle.experiments.entry` to resume an interrupted run in place.
The launcher refuses to overwrite an existing run. For Kubernetes, stage the data
on the selected node beforehand. Threshold selection uses validation only.
