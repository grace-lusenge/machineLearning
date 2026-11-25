# -*- coding: utf-8 -*-
"""
modele.py : le réseau de neurones utilisé par l'application.

C'est le même réseau que dans le TP (notebook) :
    4 entrées -> 8 neurones cachés (ReLU) -> 1 sortie (sigmoïde)
écrit uniquement avec NumPy pour que les étudiants puissent lire
tout le code, sans « boîte noire ».
"""

import csv
import numpy as np

# Ordre des colonnes d'entrée (il doit être le même partout)
COLONNES = ["heures_etude", "heures_sommeil", "taux_presence", "note_precedente"]
CIBLE = "reussite"

# Informations utilisées par l'interface (libellés, unités, limites)
CARACTERISTIQUES = {
    "heures_etude":    {"libelle": "Heures d'étude",      "unite": "h / semaine", "min": 0, "max": 40,  "pas": 0.5},
    "heures_sommeil":  {"libelle": "Heures de sommeil",   "unite": "h / nuit",    "min": 0, "max": 14,  "pas": 0.5},
    "taux_presence":   {"libelle": "Taux de présence",    "unite": "%",           "min": 0, "max": 100, "pas": 1},
    "note_precedente": {"libelle": "Note précédente",     "unite": "/ 20",        "min": 0, "max": 20,  "pas": 0.5},
}


# =====================================================================
# Chargement des données
# =====================================================================
def charger_donnees(chemin):
    """Lit le fichier CSV et renvoie X (n, 4) et y (n, 1)."""
    X, y = [], []
    with open(chemin, newline="", encoding="utf-8") as f:
        for ligne in csv.DictReader(f):
            X.append([float(ligne[c]) for c in COLONNES])
            y.append(int(ligne[CIBLE]))
    return np.array(X), np.array(y).reshape(-1, 1)


# =====================================================================
# Fonctions d'activation et de perte
# =====================================================================
def relu(z):
    return np.maximum(0, z)

def relu_derivee(z):
    return (z > 0).astype(float)

def sigmoide(z):
    return 1 / (1 + np.exp(-z))

def entropie_croisee(y_vrai, y_pred):
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return float(-np.mean(y_vrai * np.log(y_pred) + (1 - y_vrai) * np.log(1 - y_pred)))


# =====================================================================
# Le réseau de neurones
# =====================================================================
class ReseauNeurones:
    """Perceptron multicouche : entrée -> couche cachée ReLU -> sortie sigmoïde."""

    def __init__(self, n_entrees=4, n_caches=8, taux_apprentissage=0.1, n_epoques=2000, graine=0):
        self.n_entrees = n_entrees
        self.n_caches = n_caches
        self.taux = taux_apprentissage
        self.n_epoques = n_epoques
        self.rng = np.random.RandomState(graine)   # même tirage aléatoire que le TP
        self.historique_train, self.historique_test = [], []

    # ---------------- normalisation ----------------
    def _normaliser(self, X):
        # On applique TOUJOURS la moyenne et l'écart-type de l'entraînement
        return (X - self.moyenne) / self.ecart_type

    # ---------------- propagation avant ----------------
    def _propagation_avant(self, Xn):
        Z1 = Xn @ self.W1 + self.b1
        A1 = relu(Z1)
        Z2 = A1 @ self.W2 + self.b2
        A2 = sigmoide(Z2)
        return Z1, A1, A2

    # ---------------- entraînement ----------------
    def entrainer(self, X_train, y_train, X_test=None, y_test=None):
        # 1. Normalisation : paramètres calculés sur l'entraînement seulement
        self.moyenne = X_train.mean(axis=0)
        self.ecart_type = X_train.std(axis=0)
        Xn = self._normaliser(X_train)
        Xn_test = self._normaliser(X_test) if X_test is not None else None

        # 2. Initialisation « He » des poids
        self.W1 = self.rng.randn(self.n_entrees, self.n_caches) * np.sqrt(2 / self.n_entrees)
        self.b1 = np.zeros((1, self.n_caches))
        self.W2 = self.rng.randn(self.n_caches, 1) * np.sqrt(2 / self.n_caches)
        self.b2 = np.zeros((1, 1))

        m = Xn.shape[0]
        for _ in range(self.n_epoques):
            # Propagation avant
            Z1, A1, A2 = self._propagation_avant(Xn)
            self.historique_train.append(entropie_croisee(y_train, A2))

            # Rétropropagation
            dZ2 = (A2 - y_train) / m
            dW2 = A1.T @ dZ2
            db2 = dZ2.sum(axis=0, keepdims=True)
            dZ1 = (dZ2 @ self.W2.T) * relu_derivee(Z1)
            dW1 = Xn.T @ dZ1
            db1 = dZ1.sum(axis=0, keepdims=True)

            # Mise à jour des poids (descente de gradient)
            self.W2 -= self.taux * dW2
            self.b2 -= self.taux * db2
            self.W1 -= self.taux * dW1
            self.b1 -= self.taux * db1

            if Xn_test is not None:
                _, _, A2_test = self._propagation_avant(Xn_test)
                self.historique_test.append(entropie_croisee(y_test, A2_test))
        return self

    # ---------------- prédiction ----------------
    def predire_proba(self, X):
        """Probabilité de réussite pour chaque ligne de X (valeurs brutes)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        _, _, A2 = self._propagation_avant(self._normaliser(X))
        return A2.ravel()

    @property
    def n_parametres(self):
        return self.W1.size + self.b1.size + self.W2.size + self.b2.size


# =====================================================================
# Construction du modèle de l'application
# =====================================================================
def construire_modele(chemin_csv, part_test=0.2, graine=0):
    """Charge les données, sépare 80/20, entraîne le réseau et calcule les métriques."""
    X, y = charger_donnees(chemin_csv)

    reseau = ReseauNeurones(graine=graine)
    indices = reseau.rng.permutation(len(X))          # mélange des lignes
    n_train = int((1 - part_test) * len(X))
    i_train, i_test = indices[:n_train], indices[n_train:]

    reseau.entrainer(X[i_train], y[i_train], X[i_test], y[i_test])

    # Évaluation sur le test
    proba_test = reseau.predire_proba(X[i_test])
    pred = (proba_test > 0.5).astype(int)
    vrai = y[i_test].ravel()
    metriques = {
        "precision": float(np.mean(pred == vrai)),
        "vp": int(np.sum((pred == 1) & (vrai == 1))),
        "vn": int(np.sum((pred == 0) & (vrai == 0))),
        "fp": int(np.sum((pred == 1) & (vrai == 0))),
        "fn": int(np.sum((pred == 0) & (vrai == 1))),
        "n_train": int(n_train),
        "n_test": int(len(i_test)),
        "perte_finale_train": reseau.historique_train[-1],
        "perte_finale_test": reseau.historique_test[-1],
    }
    return reseau, metriques, X, y
