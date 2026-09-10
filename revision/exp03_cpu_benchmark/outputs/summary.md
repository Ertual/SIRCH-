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

## Debit reel soutenable et portee du terme temps reel

La moyenne bout en bout de 98.85 ms/frame correspond a environ 10.12 frames/s, alors qu'un flux de 25 ou 30 fps impose respectivement 40,00 ou 33,33 ms/frame. Le pipeline CPU mesure ne peut donc pas traiter exhaustivement un flux continu a 25-30 fps sans perte ni retard.

La latence normalisee ne doit pas etre confondue avec la duree d'un appel du modele : une prediction sur 20 frames prend en moyenne 1014.29 ms en inference pure. Avec `--infer-every 5`, un flux de 25 fps demanderait 5 predictions/s, soit au plus 200 ms par appel, et un flux de 30 fps en demanderait 6, soit 166,67 ms par appel. La mesure d'environ 1.01 s par appel reste trop lente, meme avec cet espacement.

Dans `main.py`, acquisition, pretraitement, prediction et affichage sont synchrones dans un seul thread. Pendant `model.predict`, aucune frame n'est lue. L'application ne possede ni file d'attente explicite, ni thread de capture, ni mecanisme explicite de saut de frames. Elle demande seulement `CAP_PROP_BUFFERSIZE=1` a OpenCV sans verifier que le pilote l'accepte. Si ce tampon est respecte, les images arrivees pendant l'inference sont abandonnees ou remplacees par la plus recente ; s'il est ignore, le tampon du pilote peut accumuler des images et produire du retard. `--infer-every 5` espace les predictions, mais ne resout pas le blocage pendant une prediction.

La revendication exacte est donc : demonstration interactive ou quasi temps reel avec echantillonnage/perte possible de frames sur ce CPU, et non traitement exhaustif garanti a 25-30 fps. Une revendication de temps reel plein debit exige une mesure directe de la boucle camera et une optimisation ou une architecture asynchrone validant au plus 40 ms/frame a 25 fps (33,33 ms/frame a 30 fps).
