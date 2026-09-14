# exp12 - Audit perceptuel hard-negative contre nouveau train

Cet audit compare les 30 videos hard-negative v2 scellees aux 2 785 videos du
nouveau train principal groupe par scene (`exp11`). Il reutilise sans modification
la methode perceptuelle de `exp10` : neuf positions temporelles, trois recadrages,
pHash 64 bits, index LSH et confirmation par correspondances temporelles ordonnees.

L'audit doit etre termine avant les quatre reentrainements groupes et avant toute
nouvelle evaluation hard-negative.

```powershell
C:\SIRCH_ENV\Scripts\python.exe revision\exp12_hard_negative_leakage_audit\audit_hard_negative_vs_grouped_train.py
```

Le verdict principal est publie dans `outputs/summary.md` et `outputs/result.json`.
