# Evaluation GRU et factoriel 2x2

Cette experience finalise les deux cellules GRU sur les memes manifestes propres que
les cellules LSTM.

Ordre obligatoire :

1. `select_threshold_k_gru.py` selectionne theta et K sur les 594 videos de
   validation, sans lire le test. Les deux GRU partagent un backbone EfficientNetB0
   verifie bit a bit, mais conservent des tetes, scores et selections distincts.
2. `lock_test_protocols.py` scelle les deux configurations et les artefacts modeles.
3. `evaluate_locked_test_grus.py` ouvre une seule fois les 592 videos du test propre
   et publie les deux evaluations finales verrouillees.
4. `analyze_factorial_2x2.py` assemble les quatre cellules LSTM/GRU x
   original/enrichi et estime l'interaction par bootstrap apparie.

Le GRU enrichi utilise exclusivement
`best_gru_augmented_clean.weights.h5`, SHA-256
`f7d3a87697d6b6b1ca36d46cdf7c1af6a499557b3e60dccc06078ff26a15956e`,
issu de l'epoch 5. Le checkpoint de l'epoch 11 est explicitement exclu.
