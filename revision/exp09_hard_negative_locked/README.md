# Experience 09 - Hard-negative verrouille

Cette experience mesure les faux positifs sur les 30 videos non violentes du corpus
hard-negative v2 scelle (15 sport, 10 danse, 5 calme).

Les quatre modeles utilisent exclusivement leur configuration selectionnee sur la
validation de 594 videos :

- LSTM original : theta 0.70, K 7 ;
- LSTM enrichi : theta 0.60, K 7, meilleurs poids de l'epoch 6 ;
- GRU original : theta 0.60, K 10 ;
- GRU enrichi : theta 0.60, K 7, meilleurs poids de l'epoch 5.

Le script verifie les SHA-256 du manifeste et des quatre modeles, confirme que les
backbones EfficientNetB0 sont identiques bit a bit, extrait les caracteristiques une
seule fois par video, puis applique chaque tete temporelle et sa regle verrouillee.

Execution :

```powershell
C:\SIRCH_ENV\Scripts\python.exe revision\exp09_hard_negative_locked\evaluate_hard_negative_locked.py
```

Les sorties reproductibles sont publiees dans `outputs/`.
