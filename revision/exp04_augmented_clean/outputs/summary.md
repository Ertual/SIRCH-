# Corpus augmente propre

Les 25 videos historiques ont ete retrouvees. Aucun nouveau sourcing n'a ete necessaire.

- Train principal nettoye : 2793
- Enrichissement historique : 25 videos non violentes
- Train augmente total : 2818
- Validation inchangee : 594
- Test principal inchange : 592
- Hard-negative v2 inchange : 30
- Doublons SHA-256 internes a l'enrichissement : 0
- Chevauchement avec train principal : 0
- Chevauchement avec validation : 0
- Chevauchement avec test principal : 0
- Chevauchement avec hard-negative v2 : 0

Les anciens dossiers test_sport et test_danse deviennent exclusivement des sources d'entrainement.
Ils ne doivent plus etre presentes comme un jeu de test. L'evaluation hard-negative utilise uniquement v2.
