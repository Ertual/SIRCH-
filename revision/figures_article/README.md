# Article figures - grouped-scene revision

These exports use the grouped-scene protocol. The four theta/K choices were selected on 597 validation videos and locked before the 597-video test was read. No figure in this directory uses the older, ungrouped 599-video test curves.

| File | Article caption / interpretation |
|---|---|
| `figure_1_architecture` | Seven operational SIRCH modules; the spatial EfficientNet-B0 and temporal LSTM stages share module 3. The dashboard's theta/K settings feed the decision module. |
| `figure_2_f1_vs_theta` | Joint theta/K F1 sweep on grouped validation only (n=597); stars mark locked choices. |
| `figure_3_training_four_models` | Training and validation loss/accuracy for all four grouped-scene retrainings; dashed lines mark best validation-loss epochs. |
| `figure_4_roc_overlay` | Four ROC curves on the same, once-evaluated grouped test (n=597). AUC is threshold-independent. |
| `figure_5_factorial_2x2` | Accuracy, F1 and violence recall for the LSTM/GRU x original/enriched design on the grouped test. |
| `figure_6_interaction_ci95` | Paired-bootstrap 95% intervals for the difference-in-differences; accuracy is the prespecified primary scale, other metrics are descriptive. |

Each numbered figure has both 300-dpi PNG and editable SVG versions. Four `training_*.png` files retain the original per-model two-panel plots.

## Locked configurations

| Model | theta | K | Best epoch |
|---|---:|---:|---:|
| LSTM / original | 0.70 | 5 | 3 |
| LSTM / enriched | 0.70 | 7 | 3 |
| GRU / original | 0.65 | 5 | 4 |
| GRU / enriched | 0.65 | 10 | 4 |

Source data: `revision/exp13_grouped_retraining/outputs/` and `revision/exp14_grouped_factorial_evaluation/outputs/`. `manifest.json` records SHA-256 provenance for every input and export.

Figure 1 follows the current `main.py` flow: acquisition, preprocessing, inference, decision, alerts, incident log, dashboard. It corrects the reviewer's six-versus-seven-step mismatch. The deployed model in `main.py` is LSTM; the GRU is an experimental alternative in the factorial study.
