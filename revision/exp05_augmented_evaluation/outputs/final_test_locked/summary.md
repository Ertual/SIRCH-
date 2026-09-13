# Evaluation finale unique du modele LSTM enrichi

Configuration verrouillee sur validation : theta=0.60, K=7, N=20, stride=5.
Aucun parametre n'a ete ajuste apres lecture du test.
Les poids utilises sont ceux de l'epoch 6, restaures par EarlyStopping.

- Videos : 592 (297 violence, 295 non-violence)
- Accuracy : 0.9088 (90.88 %)
- Balanced accuracy : 0.9087 (90.87 %)
- Precision : 0.8932 (89.32 %)
- Rappel : 0.9293 (92.93 %)
- Specificite : 0.8881 (88.81 %)
- F1 : 0.9109 (91.09 %)
- ROC-AUC : 0.9665
- PR-AUC : 0.9661
- Matrice : TN=262, FP=33, FN=21, TP=276
