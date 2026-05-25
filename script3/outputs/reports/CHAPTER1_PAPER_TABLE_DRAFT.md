# Chapter 1 Paper Table Draft

## Table A. Node-level main comparison (node_holdout, raw_residual, lambda=0.2)

| Model | MRR | Top-1 | Top-3 | Top-5 | I Top-1 | E Top-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hydraulic_inverse | 0.8383 | 0.8022 | 0.8377 | 0.9150 | 0.9962 | 0.6675 |
| lstm_graphsage_edge | 0.9147 | 0.8764 | 0.9521 | 0.9969 | 0.9962 | 0.7932 |
| tcn_graphsage_edge | 0.8965 | 0.8362 | 0.9706 | 0.9815 | 0.9811 | 0.7356 |

Suggested wording: Under the node_holdout protocol, lstm_graphsage_edge achieves the best overall localization performance and the strongest E-type generalization.

## Table B. Segment-level current best result (segment_ie_full_v1)

| Model | Task Mode | MRR | Top-1 | Top-3 | Top-5 | Type Macro-F1 | I Top-1 | E Top-1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| segment_temporal_fusion | segment | 0.7094 | 0.6246 | 0.7885 | 0.8011 | 0.3194 | 0.7910 | 0.4375 |

Suggested wording: The segment-mode pipeline has been fully closed, producing a usable formal result. Performance on E-type cases still lags behind I-type cases, leaving room for future enhancement.

## Table C. Stage summary

| Line | Status | Current conclusion |
| --- | --- | --- |
| Node mode | Completed | Main model can be fixed as lstm_graphsage_edge |
| Segment mode | Formally closed | Current best run reaches MRR 0.7094 / Top-1 0.6246 |
| Joint I/E head | Connected | Already available in segment formal run |
| Physics-refined defect definition | Deferred | Keep for later method enhancement |

## Figure suggestions

1. A grouped bar chart for node-level Top-1 by model and defect type (I vs E).
2. A grouped bar chart comparing node-level best model vs segment-level current best on MRR / Top-1 / Top-3 / Top-5.
3. A small ablation figure highlighting smoke vs full_v1 for segment mode, especially E Top-1 improvement.
