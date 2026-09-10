# Exp03 - Benchmark CPU reproductible

## Objectif

Mesurer la latence du modele original dans des conditions reproductibles et
publier la distribution, pas seulement une moyenne.

## Protocole preregistre

- Modele : `C:\SIRCH_ENV\models\sirch_model.h5`.
- GPU desactive avec `CUDA_VISIBLE_DEVICES=-1`.
- Batch : 1 sequence de 20 frames.
- Source : 100 videos du manifeste de validation, choisies avec `seed=42`,
  equilibrees en 50 violentes et 50 non violentes.
- Warm-up : 10 sequences, exclues des mesures.
- Repetitions : 3 passages des memes 100 videos, soit 300 observations.
- Inference pure : appel identique a `ViolenceInference.predict`, frames deja
  pretraitees.
- Bout en bout : ouverture du fichier, echantillonnage uniforme de 20 frames,
  redimensionnement 224 x 224, pretraitement et prediction.
- Statistiques : moyenne, ecart-type, P50, P95, P99, minimum et maximum.
- Le chargement du modele et les controles SHA-256 sont mesures a part ou exclus.

Le benchmark ne lit pas le jeu de test principal.

## Commande

```powershell
C:\SIRCH_ENV\Scripts\python.exe revision\exp03_cpu_benchmark\benchmark_cpu.py
```

## Sorties

- `outputs/latency_raw.csv`
- `outputs/latency_summary.csv`
- `outputs/benchmark_environment.json`
- `outputs/latency_percentiles.png`
- `outputs/summary.md`

## Interpretation

La latence par frame est la latence d'une sequence divisee par 20, pour rester
comparable aux rapports precedents. La latence operationnelle d'une prediction
reste la valeur par sequence.

Le resume compare aussi ce protocole a l'ancienne mesure sur seulement 5 videos
RLVS non violentes. Les warm-ups sont exclus dans les deux cas ; la nouvelle
reference est plus robuste car elle couvre 100 videos equilibrees, trois passages
et publie les mesures brutes ainsi que les percentiles.

Le resume qualifie egalement la revendication de temps reel. Le debit soutenable
est calcule a partir de la latence bout en bout et confronte aux budgets de 40 ms
par frame a 25 fps et 33,33 ms par frame a 30 fps. Le comportement de la boucle
camera de `main.py` est documente separement afin de ne pas confondre espacement
des predictions, abandon eventuel de frames par le pilote et traitement complet
du flux.
