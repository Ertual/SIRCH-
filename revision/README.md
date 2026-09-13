# Revision scientifique SIRCH

Ce dossier regroupe les experiences reproductibles executees dans l'ordre demande.
Chaque experience conserve son protocole, son script et ses sorties verifiables dans
un sous-dossier dedie.

## Experiences terminees

1. `exp01_manifests_sha256` : reconstruction et scellement SHA-256 des ensembles
   train, validation, test principal et hard-negative v2.
2. `exp02_validation_threshold_k` : selection de theta et K sur la validation
   uniquement, sans lecture du test principal.
3. `exp03_cpu_benchmark` : benchmark CPU reproductible de l'inference pure et du
   pipeline de bout en bout, avec moyenne, ecart-type, P50, P95 et P99.
4. `exp04_augmented_clean` : construction du corpus augmente propre et
   reentrainement LSTM avec restauration des meilleurs poids de l'epoch 6.
5. `exp05_augmented_evaluation` : selection validation puis evaluation finale
   verrouillee du LSTM enrichi.
6. `exp06_original_vs_augmented_stats` : McNemar exact et bootstrap apparie entre
   les deux LSTM sur le test propre.
7. `exp07_gru_augmented_clean` : entrainement de la cellule GRU enrichie avec
   restauration des meilleurs poids de l'epoch 5.
8. `exp08_gru_factorial_evaluation` : selection et evaluation verrouillees des GRU,
   puis analyse du factoriel LSTM/GRU x original/enrichi.

## Figures de revision

`generate_revision_figures.py` regenere les courbes de selection de theta, les
courbes d'entrainement, les courbes ROC et la figure d'interaction factorielle.
Chaque fichier est ecrit dans le dossier `outputs/` de l'experience correspondante
et son SHA-256 est enregistre dans le `figure_manifest.json` local. L'index agrege
de toutes les figures reste disponible dans `revision/figure_manifest.json`.

## Regles de lecture

- Les modeles et les videos sources restent hors de Git.
- Les manifestes contiennent les chemins, classes, tailles et empreintes SHA-256.
- Les fichiers `README.md` de chaque experience decrivent le protocole exact.
- Les sorties publiees sont celles presentes dans les dossiers `outputs`.
- Le jeu de test principal n'a ete lu ni pour la selection de theta/K ni pour le
  benchmark CPU.

## Point de controle avant l'experience 4

Avant tout reentrainement augmente, rechercher la liste historique exacte des 25
videos d'enrichissement dans Git, les anciens notebooks et les journaux. Si elle est
introuvable, tout nouveau lot devra etre prouve disjoint par SHA-256 du corpus
principal et des 30 videos hard-negative v2 deja scellees par l'experience 1.

Ce controle est maintenant realise dans `exp04_augmented_clean`. Les 25 videos
historiques ont ete retrouvees; aucun remplacement n'a ete necessaire.
