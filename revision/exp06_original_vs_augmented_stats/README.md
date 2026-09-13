# Comparaison statistique finale

`compare_locked_models.py` refuse de demarrer tant que les evaluations finales verrouillees
du modele original et du modele LSTM enrichi ne sont pas toutes deux presentes.

Il verifie les 592 videos par hash SHA-256, calcule le test exact de McNemar sur les
predictions appariees et produit des intervalles de confiance a 95 % par bootstrap apparie
avec 10 000 repetitions et la graine 42.
