# Splits groupes par scene/source

Aucun entrainement n'a ete lance. Cette etape s'arrete aux manifestes scelles,
conformement a la demande de validation prealable.

## Revue des labels opposes

- Paires examinees visuellement: 90
- Videos uniques examinees: 129
- Corrections de label: 5 (toutes violence -> non_violence)
- Paires devenues concordantes apres correction: 6
- Paires restant opposees car segments temporels distincts: 84
- Preuve: 17 planches a 9 instants et 5 planches agrandies a 24 instants.

| Cas | Video | Avant | Apres | Motif visuel |
|---|---|---:|---:|---|
| V096 | `train/Fight/rwf_train_fight_0249.avi` | 1 | 0 | Hall d'hotel: passants et bagages; aucun coup, contact agressif, poursuite ou menace visible sur 24 instants. |
| V102 | `train/Fight/rwf_train_fight_0509.avi` | 1 | 0 | Voiture garee seule pendant tout le clip; aucun individu ni action violente visible sur 24 instants. |
| V109 | `train/Fight/rwf_train_fight_0322.avi` | 1 | 0 | Fin de scene: les personnes se dispersent et marchent; aucun contact agressif visible dans ce segment sur 24 instants. |
| V113 | `val/Fight/rwf_val_fight_0042.avi` | 1 | 0 | Parking lointain: des pietons marchent; aucun contact ni geste agressif visible sur 24 instants. |
| V116 | `train/Fight/rwf_train_fight_0369.avi` | 1 | 0 | Homme seul pres d'une cloture; aucun adversaire, coup ou action violente visible sur 24 instants. |

## Nouveaux splits principaux

| Split | Videos | Proportion | Non-violence | Violence | Entrees | Sorties |
|---|---:|---:|---:|---:|---:|---:|
| train | 2785 | 0.69992460 | 1399 | 1386 | 122 | 130 |
| validation | 597 | 0.15003770 | 300 | 297 | 76 | 73 |
| test | 597 | 0.15003770 | 300 | 297 | 81 | 76 |

Videos ayant change de split: **279**.

## Integrite

- Groupes perceptuels de base: 191
- Groupes finaux, singletons compris: 3503
- Groupes finaux non singletons: 204
- Violations groupe/split: 0
- Violations SHA-256 exact/split: 0
- Enrichissement historique: 25 videos train-only, aucun hash exact ni groupe perceptuel commun avec le corpus principal.
- Affectation: tailles et comptes de classes exacts; minimisation du nombre de videos deplacees.

## Scellement

- `train_manifest_grouped.csv`: `55c83d2b2546c09700bfc208f5b08cc915755359dea98d58aa746c00ea9453f8` (2785 lignes)
- `validation_manifest_grouped.csv`: `d4b3f4a41c4d118a99abd70e88943f7886970695353369259ec3ea506e3964e5` (597 lignes)
- `test_manifest_grouped.csv`: `a37aa9cba85d28ac315c1ea813c902330a3e99525042c13012023914e0a9887a` (597 lignes)
- `augmented_train_manifest_grouped.csv`: `287a299b9c47808a9dda2ce51877e1f8f1b14dc00c919a3f8fdf3c446abd3b2f` (2810 lignes)

## Arret controle

Les nouveaux manifestes n'ont ete consommes par aucun entrainement. Les quatre
reentrainements restent bloques dans l'attente de la validation de Tracy.
