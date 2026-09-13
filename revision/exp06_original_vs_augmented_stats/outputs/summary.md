# Comparaison finale appariee : original contre enrichi

Les deux configurations ont ete verrouillees sur validation puis appliquees une seule fois aux memes 592 videos de test.
La sequence des 592 lignes a ete verifiee position par position par SHA-256.
Le test contient 591 contenus SHA-256 uniques et 1 groupe de doublon interne, sans fuite entre splits.

## McNemar exact

- Corrects par les deux : 504
- Original correct, enrichi faux (b) : 17
- Original faux, enrichi correct (c) : 34
- Faux pour les deux : 37
- p bilaterale exacte : 0.0240929
- Difference significative a 5 % : oui

## Estimations et IC 95 % apparies

| Metrique | Original | Enrichi | Delta enrichi-original [IC 95 %] |
|---|---:|---:|---:|
| accuracy | 0.8801 | 0.9088 | +0.0287 [+0.0051; +0.0524] |
| balanced_accuracy | 0.8799 | 0.9087 | +0.0288 [+0.0052; +0.0527] |
| precision | 0.8509 | 0.8932 | +0.0423 [+0.0127; +0.0731] |
| recall | 0.9226 | 0.9293 | +0.0067 [-0.0230; +0.0365] |
| specificity | 0.8373 | 0.8881 | +0.0508 [+0.0143; +0.0883] |
| f1 | 0.8853 | 0.9109 | +0.0256 [+0.0031; +0.0485] |
| roc_auc | 0.9527 | 0.9665 | +0.0138 [+0.0043; +0.0240] |
| pr_auc | 0.9509 | 0.9661 | +0.0153 [+0.0050; +0.0266] |
