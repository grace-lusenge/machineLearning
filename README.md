# RéussiteIA : application Flask de prédiction de réussite

Application web qui utilise le **réseau de neurones du TP** (4 entrées → 8 neurones cachés ReLU → 1 sortie sigmoïde, écrit avec NumPy) pour prédire la probabilité qu'un étudiant réussisse son examen.

## Installation et lancement

```bash
cd app_reussite
pip install -r requirements.txt
python app.py
```

Puis ouvrir **http://127.0.0.1:5000** dans le navigateur.

Au démarrage, l'application entraîne le réseau sur `data/reussite_etudiants.csv` (moins d'une seconde) et affiche sa précision sur le test (96,7 %).

## Les pages

| Page | Ce qu'elle fait |
|---|---|
| **Prédire** (`/`) | Curseurs pour régler le profil d'un étudiant. La probabilité se met à jour en direct, avec des conseils (« dormir 7 h 30 → +82 points ») et un avertissement si une valeur sort de la plage vue à l'entraînement. |
| **Prédiction en lot** (`/lot`) | Envoi d'un fichier CSV de plusieurs étudiants, tableau des résultats, téléchargement des prédictions en CSV. Un fichier d'exemple est fourni. |
| **Le modèle** (`/modele`) | Architecture du réseau, précision, matrice de confusion, courbe d'apprentissage. |
| **Les données** (`/donnees`) | Statistiques par colonne, nuage de points étude/sommeil, tableau filtrable des 150 étudiants. |

Il existe aussi une **API JSON** :

```bash
curl -X POST http://127.0.0.1:5000/api/predire \
     -H "Content-Type: application/json" \
     -d '{"heures_etude": 15, "heures_sommeil": 7.5, "taux_presence": 90, "note_precedente": 14}'
```

Réponse : `{"probabilite": 100.0, "verdict": "Réussite", "niveau": "Réussite probable", "conseils": [...], ...}`

## Structure du projet

```
app_reussite/
├── app.py                  <- application Flask (pages, API, lecture des CSV)
├── modele.py               <- le réseau de neurones NumPy (le même que dans le TP)
├── test_app.py             <- tests automatiques (python -m pytest -q)
├── requirements.txt
├── data/
│   └── reussite_etudiants.csv
├── exemples/
│   └── nouveaux_etudiants.csv   <- fichier d'exemple pour la prédiction en lot
├── templates/              <- pages HTML (Jinja2)
│   ├── base.html, index.html, lot.html, modele.html, donnees.html, erreur.html
└── static/
    ├── css/style.css
    └── js/app.js           <- curseurs, prédiction en direct, téléchargement CSV
```

## Comment ça marche

1. `modele.py` lit le CSV, sépare 80 % entraînement / 20 % test, normalise les données et entraîne le réseau par rétropropagation (2 000 époques, taux d'apprentissage 0,1).
2. `app.py` garde le modèle entraîné en mémoire. À chaque prédiction, les valeurs sont normalisées **avec la moyenne et l'écart-type de l'entraînement**, puis passent dans le réseau.
3. Dans le navigateur, `app.js` envoie les valeurs des curseurs à `/api/predire` et met à jour la jauge sans recharger la page. Sans JavaScript, le bouton « Calculer la prédiction » envoie le formulaire classiquement.

## Idées d'exercices

- Changer `n_caches` ou `taux_apprentissage` dans `modele.py`, relancer, et comparer la précision affichée sur la page « Le modèle ».
- Ajouter un nouveau conseil dans la fonction `conseils()` de `app.py`.
- Ajouter une route `/api/modele` qui renvoie les métriques en JSON.

> Les données sont fictives : cette application sert à apprendre, pas à juger de vrais étudiants.
>
> `python app.py` lance le serveur de développement de Flask, prévu pour un usage en classe ou en local. Pour une mise en ligne, utiliser un serveur WSGI (par exemple `gunicorn app:app`).
