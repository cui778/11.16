# Chapter 4 Formal Window-Length Sensitivity Audit

- dataset: `ie420_plus_normal20_v1`
- model: `hydraulic_inverse_deepattn`
- layout: `degree_N25`
- feature set: `raw_plus_residual`
- lambda_loc: `0.5`
- split: `scenario`
- seed: `42`
- stride: `6` samples = `1 h`
- varied field only: `sequence_length`

The 6 h row reuses the existing formal Chapter 4 main-model checkpoint.
Window-level detection and localization metrics are the formal sensitivity evidence.
Predicted-segment timeline fields are exploratory audit fields only: the simple first-consecutive-active-segment postprocessor is sensitive to early alarms and is not used as headline evidence for temporal boundary recovery.
