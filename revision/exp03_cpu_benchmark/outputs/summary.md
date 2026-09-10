# Benchmark CPU SIRCH original

Observations : 300 apres 10 warm-ups.

| Protocole | Moyenne ms/frame | P50 | P95 | P99 |
|---|---:|---:|---:|---:|
| Inference pure | 50.71 | 49.88 | 56.51 | 60.81 |
| Bout en bout | 98.85 | 76.71 | 199.47 | 231.51 |

Le chargement du modele et les controles SHA-256 sont exclus des latences.
Le jeu de test principal n'a pas ete lu.

## Ecart avec les chiffres publies auparavant

| Mesure | Ancien protocole | Nouveau protocole | Ecart |
|---|---:|---:|---:|
| Inference pure | 28.87 | 50.71 | +21.84 ms/frame (75.7 %) |
| Bout en bout | 47.33 | 98.85 | +51.52 ms/frame (108.9 %) |

L'ancien benchmark (`phase3_results/model_complexity.txt`) utilisait seulement 5 videos RLVS non violentes, soit 5 mesures de sequence et 100 frames au total, en un seul passage. Le nouveau benchmark utilise 100 videos de validation equilibrees (50 violentes et 50 non violentes, RLVS et RWF-2000), trois passages et 300 mesures.

Le warm-up n'explique pas l'ecart : il etait deja exclu de l'ancienne mesure et les 10 warm-ups du nouveau protocole sont egalement exclus. Le nombre de 300 mesures ne ralentit pas mathematiquement une prediction ; il expose mieux la variabilite et la charge soutenue du CPU.

La variation entre repetitions le confirme : inference pure 53.39, 49.94, 48.81 ms/frame ; bout en bout 101.88, 98.07, 96.59 ms/frame. Les passages deviennent ici legerement plus rapides, ce qui est compatible avec la stabilisation des caches TensorFlow, systeme et disque. Le pipeline complet ajoute aussi la variabilite des codecs, de la lecture disque et du redimensionnement sur un corpus plus heterogene.

Les deux chiffres ne sont donc pas directement comparables. L'ancien resultat est une petite mesure ponctuelle favorable ; le nouveau resultat, avec distribution P50/P95/P99 et mesures brutes, est la reference reproductible a retenir.
