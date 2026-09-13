# Factoriel SIRCH 2x2 sur test propre

Les quatre configurations ont leur propre theta/K choisi sur validation et sont comparees sur les memes 592 videos de test.

| Architecture | Enrichissement | theta | K | Accuracy | Precision | Rappel | Specificite | F1 | ROC-AUC | TN/FP/FN/TP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| LSTM | original | 0.70 | 7 | 0.8801 | 0.8509 | 0.9226 | 0.8373 | 0.8853 | 0.9527 | 247/48/23/274 |
| LSTM | enrichi | 0.60 | 7 | 0.9088 | 0.8932 | 0.9293 | 0.8881 | 0.9109 | 0.9665 | 262/33/21/276 |
| GRU | original | 0.60 | 10 | 0.8767 | 0.8373 | 0.9360 | 0.8169 | 0.8839 | 0.9553 | 241/54/19/278 |
| GRU | enrichi | 0.60 | 7 | 0.8818 | 0.8388 | 0.9461 | 0.8169 | 0.8892 | 0.9572 | 241/54/16/281 |

## Interaction primaire sur l'accuracy

- Effet enrichissement avec LSTM : +0.0287
- Effet enrichissement avec GRU : +0.0051
- Difference des differences : -0.0236 [IC 95 % -0.0507; +0.0034]
- Test exact par permutation de signes appariee : p=0.108957
- Conclusion : Aucune interaction statistiquement detectee a 5 %.

L'interaction est definie sur l'echelle additive des probabilites de classification correcte.
