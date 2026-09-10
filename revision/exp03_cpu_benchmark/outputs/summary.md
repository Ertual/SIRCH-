# Benchmark CPU SIRCH original

Observations : 300 apres 10 warm-ups.

| Protocole | Moyenne ms/frame | P50 | P95 | P99 |
|---|---:|---:|---:|---:|
| Inference pure | 49.47 | 49.18 | 63.01 | 71.20 |
| Bout en bout | 85.96 | 72.49 | 173.86 | 226.81 |

Le chargement du modele et les controles SHA-256 sont exclus des latences.
Le jeu de test principal n'a pas ete lu.
