## Current Visualization Output Status

Previously generated visualization images were moved to:
- `_legacy_generated_20260326`

Reason:
- older generated images may reflect invalidated monitor-only datasets or legacy report directories

Current root contents under `visualizations` should be interpreted as:
- Python scripts = active generation entrypoints
- `_legacy_generated_20260326` = archived historical image outputs

Regenerate visualization outputs only from the current clean main-line datasets and reports.
