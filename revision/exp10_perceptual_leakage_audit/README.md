# Experience 10 - Audit perceptuel anti-fuite

Cet audit complete le controle SHA-256 exact par une recherche de quasi-doublons
entre le train augmente (2 793 videos principales et 25 enrichissements), la
validation (594) et le test propre (592).

Pour chaque video, le script extrait neuf frames reparties entre 5 % et 95 % de la
duree. Il calcule un pHash 64 bits sur l'image complete et sur deux recadrages
centraux (10 % et 20 %), puis recherche les paires inter-splits ayant au moins une
distance de Hamming inferieure ou egale a 8. Une paire n'est signalee comme
quasi-doublon qu'apres confirmation par plusieurs correspondances temporelles
ordonnees, ou par un identifiant de source/nom exact.

La duree, la resolution, le nombre de frames et l'identifiant de source disponible
sont aussi conserves pour l'examen des candidats. Cette methode vise les
re-encodages, petits recadrages et decoupes legerement decalees; elle ne garantit
pas la detection d'un fragment commun tres court absent des neuf frames.

Execution complete :

```powershell
C:\SIRCH_ENV\Scripts\python.exe revision\exp10_perceptual_leakage_audit\audit_perceptual_leakage.py
```

Les signatures, candidats, conclusions et empreintes des artefacts sont publies
dans `outputs/`.

Un controle visuel cible de 24 paires (cas forts et cas limites) est documente dans
`outputs/manual_visual_review.json` et dans trois planches de contact. Il soutient
la classification automatique sans etre presente comme une estimation aleatoire
de sa precision sur toutes les paires.
