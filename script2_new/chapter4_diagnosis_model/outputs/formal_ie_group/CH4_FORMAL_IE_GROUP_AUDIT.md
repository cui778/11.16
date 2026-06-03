# Chapter 4 Formal I/E Grouped Localization Audit

- dataset: `ie420_plus_normal20_v1`
- model: `hydraulic_inverse_deepattn`
- layout: `degree_N25`
- feature set: `raw_plus_residual`
- lambda_loc: `0.5`
- split: `scenario`
- seeds: `7, 42, 123`

I/E grouping is a post-hoc stratification of the same localization task.
It is not a defect-type classification task and Type Accuracy is not reported.
Formal event-level localization metrics aggregate node scores over true active windows.
Integrated-scene metrics use predicted active segments and remain diagnostic audit fields because they additionally depend on the temporal postprocessor.
