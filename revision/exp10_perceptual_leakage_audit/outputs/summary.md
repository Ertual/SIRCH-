# Audit perceptuel anti-fuite

Audit complet de 4004 videos et 3693796 paires inter-splits possibles.
Le train audite inclut les 2 793 videos principales et les 25 videos d'enrichissement.

**Verdict : 626 paire(s) probable(s) ou possible(s) a examiner.**

- Chevauchements SHA-256 exacts : 0
- Groupes SHA-256 internes aux splits (hors fuite) : 13
- Paires candidates pHash analysees : 1356
- Quasi-doublons probables : 410
- Quasi-doublons possibles : 216
- Similarites de frame isolees rejetees : 730
- Videos test avec un quasi-doublon probable dans le train : 97
- Videos validation avec un quasi-doublon probable dans le train : 98
- Paires probables avec labels opposes : 90
- Videos avec decodage incomplet : 1
- Controle visuel cible : 23/24 paires compatibles avec une meme source/scene, 1 inconclusive, 0 contradiction

La methode utilise neuf positions temporelles, trois cadrages par frame (0, 10 et 20 %),
un pHash 64 bits et une confirmation par correspondances temporelles ordonnees. Les
metadonnees de source, duree, resolution et nom sont conservees dans les sorties.

Limite : un fragment commun tres court entre deux videos longues peut tomber entre les
positions echantillonnees; un recadrage severe avec surimpression peut aussi echapper au pHash.
