# Resume des manifestes SIRCH

Generation UTC : 2026-09-10T07:00:51.801758+00:00

| Ensemble | Total | Violence | Non-violence | SHA-256 du manifeste |
|---|---:|---:|---:|---|
| train | 2793 | 1392 | 1401 | `df855f1d1e0daabc1aa72c897ff2b4b0c09bba4a09cb7110bbb5f72484224767` |
| validation | 594 | 296 | 298 | `933a26378a88236beafeb51fc91b53c7abab5185aba9d97daa0b68dd2310e397` |
| test | 592 | 297 | 295 | `13bfda2224edb723a75be0deb4bd067d37fb7c481830a00cd11284485994cb1f` |
| hard_negative_v2 | 30 | 0 | 30 | `4b9c6c135ec3869e7b109f1aa5a695b3a544368b07ada753919ca226e9c76dba` |

Le test historique reconstruit correspondait exactement aux 599 videos publiees avant deduplication.
Groupes SHA-256 inter-splits detectes avant correction : 12.
Lignes dupliquees retirees des manifestes : 12.
Groupes SHA-256 inter-splits apres correction : 0.
Chevauchements hard-negative v2 / corpus principal : 0.
Conflits d'etiquettes avant/apres correction : 1/0.

Regle canonique : conserver train, sinon validation, sinon test. Aucun fichier video source n'a ete supprime.
Le conflit V_504.mp4 / NV_226.mp4 est un match de tennis : etiquette canonique corrigee en non_violence.

| Groupe | SHA-256 | Splits d'origine | Occurrence conservee | Occurrence retiree |
|---:|---|---|---|---|
| 1 | `3f9cc7f69f5d74f305d3f56be41b0cb9b34e4a42b08c7ea8e5517f6cbdcaf054` | train<->validation | train:train/Fight/rwf_train_fight_0531.avi | validation:train/Fight/rwf_train_fight_0431.avi |
| 2 | `524d31e12b2a4ddec8bf5a4728008768db94578525ab0ca54a1afea69a4f9752` | train<->validation | train:NonViolence/NV_853.mp4 | validation:NonViolence/NV_852.mp4 |
| 3 | `5561ba5284a1049ef6c937e9e2aea621936605c9028f059d7da875e989784124` | train<->validation | train:train/Fight/rwf_train_fight_0790.avi | validation:train/Fight/rwf_train_fight_0791.avi |
| 4 | `6dcca31f1b764aba418928b6b8744721e060adb0b36af54ad86e60aa2bfb9448` | train<->test | train:train/Fight/rwf_train_fight_0532.avi | test:train/Fight/rwf_train_fight_0432.avi |
| 5 | `824edfa52b7b6fecd85754be7f841cdd5cecb7519103623d4c17f2a325039612` | train<->test | train:Violence/V_504.mp4 | test:NonViolence/NV_226.mp4 |
| 6 | `961e50c0e08d60751610037727aa09fa4258c2f2be30c6e98644e2b2678db001` | train<->test | train:train/NonFight/rwf_train_nonfight_0071.avi | test:train/NonFight/rwf_train_nonfight_0761.avi |
| 7 | `992382c47f49ffe960ab3f609d48d4e2049490e44c048b4ef6c10f843e8fefaf` | train<->test | train:NonViolence/NV_27.mp4 | test:NonViolence/NV_20.mp4 |
| 8 | `a267d4f80b188d8bf175a61e1699f28733c4c180be8f9c0411d273e2e1454f6b` | validation<->test | validation:Violence/V_447.mp4 | test:Violence/V_431.mp4 |
| 9 | `a85a93055d0601cf6ab7dfd9dd8e6eb5e701b65baa7deb3bcaaaf56cdf1d04e4` | train<->test | train:NonViolence/NV_297.mp4 | test:NonViolence/NV_298.mp4 |
| 10 | `b4899c39e0d4575dbb277f3cab6a56937c57b72503b7c06278b7203105592de2` | train<->test | train:NonViolence/NV_25.mp4 | test:NonViolence/NV_26.mp4 |
| 11 | `c344e296d5084b7caee6a296b4a7a8773d0555f302336c9420d7ea05ca8271f9` | train<->validation | train:train/NonFight/rwf_train_nonfight_0510.avi | validation:val/NonFight/rwf_val_nonfight_0049.avi |
| 12 | `fcff55a63ffb5f6908aa18db2afe3c44cc6b0f62d010c585b698f54373c2f5a8` | train<->validation | train:Violence/V_163.mp4 | validation:Violence/V_169.mp4 |

Voir `duplicate_resolution.csv` pour les etiquettes, actions et justifications detaillees.
`cross_split_duplicates.csv` est vide apres resolution (en-tete uniquement).
