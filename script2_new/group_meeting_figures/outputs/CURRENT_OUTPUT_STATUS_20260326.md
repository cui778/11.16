## Current Figure Output Status

All previously generated PPT/group-meeting figure images have been moved to:
- `_legacy_invalidated_20260326`

Reason:
- historical figures were produced under mixed data lineages and older result directories
- leaving them in the root output folder would make it too easy to reuse outdated visuals

Current rule:
- regenerate figures only after confirming they point to the clean main-line data and report files

This folder is intentionally left without default figure PNGs in the root.
