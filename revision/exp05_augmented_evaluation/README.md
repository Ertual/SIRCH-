# Evaluation verrouillee du modele LSTM enrichi

Ordre obligatoire :

1. `select_threshold_k_augmented.py` selectionne theta et K sur les 594 videos de validation.
2. `lock_test_protocol.py` scelle la configuration et les empreintes avant le test.
3. `evaluate_locked_test_augmented.py` ouvre et evalue une seule fois les 592 videos du test propre.

Le modele est toujours reconstruit avec l'architecture d'entrainement, puis charge depuis
`best_lstm_augmented_clean.weights.h5`, dont l'empreinte SHA-256 attendue est
`4ccfe5bb3838cd829fa5e4eaba0332128d14e52380509eee2d4083c321aeaa13`.
Le checkpoint de l'epoch 12 est explicitement exclu.
