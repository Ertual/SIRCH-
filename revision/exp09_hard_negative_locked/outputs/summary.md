# Faux positifs hard-negative - configurations verrouillees

Evaluation categorielle des 30 videos non violentes du corpus v2 scelle.
Aucun theta ni K n'a ete reselectionne sur ce corpus.

| Modele | Categorie | Faux positifs | Taux |
|---|---:|---:|---:|
| lstm_original | sport | 8/15 | 53.33% |
| lstm_original | danse | 7/10 | 70.00% |
| lstm_original | calme | 4/5 | 80.00% |
| lstm_original | overall | 19/30 | 63.33% |
| lstm_augmented | sport | 6/15 | 40.00% |
| lstm_augmented | danse | 5/10 | 50.00% |
| lstm_augmented | calme | 3/5 | 60.00% |
| lstm_augmented | overall | 14/30 | 46.67% |
| gru_original | sport | 6/15 | 40.00% |
| gru_original | danse | 9/10 | 90.00% |
| gru_original | calme | 4/5 | 80.00% |
| gru_original | overall | 19/30 | 63.33% |
| gru_augmented | sport | 8/15 | 53.33% |
| gru_augmented | danse | 9/10 | 90.00% |
| gru_augmented | calme | 4/5 | 80.00% |
| gru_augmented | overall | 21/30 | 70.00% |

Les predictions individuelles et les scores de decision sont dans
`hard_negative_predictions.csv`.
