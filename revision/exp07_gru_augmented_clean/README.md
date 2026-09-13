# Case factorielle GRU + enrichi

Cette experience complete le factoriel en remplacant uniquement la cellule LSTM(256)
par GRU(256). Le backbone EfficientNetB0, les couches Dense, les dropouts, le corpus
augmente scelle, la validation propre et les hyperparametres d'entrainement restent
identiques a l'experience LSTM enrichie.

Le jeu de test n'est jamais lu pendant l'entrainement. Chaque epoch produit un checkpoint,
les meilleurs poids sont selectionnes sur `val_loss`, et EarlyStopping restaure ces poids.
