# Grouped 2x2 factorial

| Architecture | Enrichment | theta | K | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC | TN/FP/FN/TP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| LSTM | original | 0.70 | 5 | 0.8610 | 0.8204 | 0.9226 | 0.8000 | 0.8685 | 0.9495 | 240/60/23/274 |
| LSTM | enriched | 0.70 | 7 | 0.8626 | 0.8228 | 0.9226 | 0.8033 | 0.8698 | 0.9557 | 241/59/23/274 |
| GRU | original | 0.65 | 5 | 0.8727 | 0.8486 | 0.9057 | 0.8400 | 0.8762 | 0.9510 | 252/48/28/269 |
| GRU | enriched | 0.65 | 10 | 0.8677 | 0.8733 | 0.8586 | 0.8767 | 0.8659 | 0.9453 | 263/37/42/255 |

## Accuracy interaction

- LSTM enrichment effect: +0.0017
- GRU enrichment effect: -0.0050
- Difference-in-differences: -0.0067 [95% CI -0.0352; +0.0218]
- Exact paired sign-flip p-value: 0.731686
- Conclusion: No interaction detected at 5%.
