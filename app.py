# -*- coding: utf-8 -*-
"""
app.py : application web Flask « Prédiction de réussite ».

Lancement :
    pip install -r requirements.txt
    python app.py
puis ouvrir http://127.0.0.1:5000 dans le navigateur.

Au démarrage, l'application entraîne le réseau de neurones du TP
sur data/reussite_etudiants.csv (moins d'une seconde), puis l'utilise
pour toutes les prédictions.
"""

import csv
import io
import os

import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory

from modele import CARACTERISTIQUES, COLONNES, construire_modele

DOSSIER = os.path.dirname(os.path.abspath(__file__))
CHEMIN_DONNEES = os.path.join(DOSSIER, "data", "reussite_etudiants.csv")
MAX_LIGNES_LOT = 1000

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024   # fichiers CSV de 1 Mo maximum

# ---------------------------------------------------------------------
# Entraînement du modèle au démarrage
# ---------------------------------------------------------------------
RESEAU, METRIQUES, X_DONNEES, Y_DONNEES = construire_modele(CHEMIN_DONNEES)
# Plage de chaque colonne dans les données d'entraînement (pour avertir
# l'utilisateur quand il sort de ce que le modèle a vu)
PLAGES = {c: (float(X_DONNEES[:, i].min()), float(X_DONNEES[:, i].max())) for i, c in enumerate(COLONNES)}

PROFILS = [
    {"nom": "Sérieux et reposé", "valeurs": [15, 7.5, 90, 14]},
    {"nom": "Manque de sommeil", "valeurs": [15, 4.5, 90, 14]},
    {"nom": "Peu présent", "valeurs": [2, 8, 50, 7]},
    {"nom": "Cas limite", "valeurs": [8, 6.5, 70, 10]},
]


# =====================================================================
# Fonctions utilitaires
# =====================================================================
@app.template_filter("fr")
def format_fr(valeur, decimales=1):
    """12.0 -> '12' ; 7.5 -> '7,5' (format français, sans zéro inutile)."""
    texte = f"{float(valeur):.{decimales}f}"
    if "." in texte:
        texte = texte.rstrip("0").rstrip(".")
    return texte.replace(".", ",")

def nombre(texte):
    """Convertit '7,5' ou '7.5' en float. Lève ValueError si impossible."""
    return float(str(texte).strip().replace(",", "."))


def lire_valeurs(source):
    """Lit et valide les 4 caractéristiques depuis un dictionnaire."""
    valeurs, erreurs = {}, []
    for c in COLONNES:
        info = CARACTERISTIQUES[c]
        brut = source.get(c, "")
        try:
            v = nombre(brut)
        except (ValueError, TypeError):
            erreurs.append(f"{info['libelle']} : valeur manquante ou invalide.")
            continue
        if not np.isfinite(v) or not (info["min"] <= v <= info["max"]):
            erreurs.append(f"{info['libelle']} doit être entre {info['min']} et {info['max']} (valeur reçue : {format_fr(v)}).")
            continue
        valeurs[c] = v
    return valeurs, erreurs


def niveau(proba):
    if proba >= 0.7:
        return "Réussite probable", "succes"
    if proba >= 0.5:
        return "Réussite possible", "succes-leger"
    if proba >= 0.3:
        return "Risque d'échec", "risque"
    return "Échec probable", "echec"


def proba_de(valeurs):
    return float(RESEAU.predire_proba([[valeurs[c] for c in COLONNES]])[0])


def conseils(valeurs, proba):
    """Teste quelques changements réalistes et garde ceux qui aident."""
    essais = []
    if abs(valeurs["heures_sommeil"] - 7.5) > 0.75:
        essais.append(("Dormir environ 7 h 30 par nuit", {"heures_sommeil": 7.5}))
    if valeurs["heures_etude"] < 20:
        essais.append(("Étudier 3 h de plus par semaine", {"heures_etude": min(valeurs["heures_etude"] + 3, 20)}))
    if valeurs["taux_presence"] < 100:
        essais.append(("Être plus présent en cours (+10 points)", {"taux_presence": min(valeurs["taux_presence"] + 10, 100)}))

    resultats = []
    for texte, changement in essais:
        apres = proba_de({**valeurs, **changement})
        if apres - proba >= 0.03:
            resultats.append({"texte": texte, "apres": round(apres * 100), "gain": round((apres - proba) * 100)})
    return sorted(resultats, key=lambda r: -r["gain"])


def avertissements(valeurs):
    msgs = []
    for c, v in valeurs.items():
        lo, hi = PLAGES[c]
        if v < lo or v > hi:
            info = CARACTERISTIQUES[c]
            msgs.append(f"{info['libelle']} ({format_fr(v)}) est en dehors des valeurs vues à l'entraînement "
                        f"({format_fr(lo)} à {format_fr(hi)}) : la prédiction est moins fiable.")
    return msgs


def analyser(valeurs):
    """Prédiction complète pour un étudiant."""
    proba = proba_de(valeurs)
    libelle, classe = niveau(proba)
    return {
        "probabilite": round(proba * 100, 1),
        "verdict": "Réussite" if proba > 0.5 else "Échec",
        "niveau": libelle,
        "classe": classe,
        "conseils": conseils(valeurs, proba),
        "avertissements": avertissements(valeurs),
    }


def lire_csv_lot(contenu):
    """Lit un CSV envoyé par l'utilisateur. Renvoie (lignes valides, erreurs)."""
    texte = contenu.decode("utf-8-sig", errors="replace")
    try:
        dialecte = csv.Sniffer().sniff(texte[:2048], delimiters=",;\t")
        sep = dialecte.delimiter
    except csv.Error:
        sep = ","
    lecteur = csv.DictReader(io.StringIO(texte), delimiter=sep)
    if not lecteur.fieldnames:
        return [], ["Le fichier est vide."]
    entetes = {h.strip().lower(): h for h in lecteur.fieldnames if h}
    manquantes = [c for c in COLONNES if c not in entetes]
    if manquantes:
        return [], ["Colonnes manquantes : " + ", ".join(manquantes) +
                    ". Le fichier doit contenir : " + ", ".join(COLONNES) + "."]
    # Une colonne d'identifiant facultative (nom, id, etudiant...)
    col_id = next((entetes[k] for k in ("nom", "id", "etudiant", "étudiant", "identifiant") if k in entetes), None)

    lignes, erreurs = [], []
    for n, ligne in enumerate(lecteur, start=2):     # ligne 1 = en-têtes
        if n - 1 > MAX_LIGNES_LOT:
            erreurs.append(f"Seules les {MAX_LIGNES_LOT} premières lignes ont été traitées.")
            break
        if not any((v or "").strip() for v in ligne.values() if isinstance(v, str)):
            continue                                   # ligne vide
        valeurs, errs = lire_valeurs({c: ligne.get(entetes[c], "") for c in COLONNES})
        if errs:
            erreurs.append(f"Ligne {n} ignorée : " + " ".join(errs))
            continue
        ident = (ligne.get(col_id) or "").strip() if col_id else ""
        lignes.append({"id": ident or f"Étudiant {len(lignes) + 1}", "valeurs": valeurs})
    return lignes, erreurs


def points_courbe(valeurs, largeur, hauteur, y_max, pas=10):
    """Transforme une liste de pertes en points SVG 'x,y x,y ...'."""
    v = valeurs[::pas] + [valeurs[-1]]
    n = len(v) - 1
    return " ".join(f"{i / n * largeur:.1f},{hauteur - min(val, y_max) / y_max * hauteur:.1f}" for i, val in enumerate(v))


# =====================================================================
# Pages
# =====================================================================
@app.route("/", methods=["GET", "POST"])
def accueil():
    valeurs = {c: v for c, v in zip(COLONNES, PROFILS[3]["valeurs"])}
    resultat, erreurs = None, []
    if request.method == "POST":
        lues, erreurs = lire_valeurs(request.form)
        valeurs.update(lues)
        if not erreurs:
            resultat = analyser(valeurs)
    else:
        resultat = analyser(valeurs)
    return render_template("index.html", caracteristiques=CARACTERISTIQUES, colonnes=COLONNES,
                           valeurs=valeurs, resultat=resultat, erreurs=erreurs, profils=PROFILS,
                           metriques=METRIQUES, page="accueil")


@app.route("/api/predire", methods=["POST"])
def api_predire():
    donnees = request.get_json(silent=True)
    if not isinstance(donnees, dict):
        return jsonify({"erreurs": ["Le corps de la requête doit être un objet JSON."]}), 400
    valeurs, erreurs = lire_valeurs(donnees)
    if erreurs:
        return jsonify({"erreurs": erreurs}), 400
    return jsonify(analyser(valeurs))


@app.route("/lot", methods=["GET", "POST"])
def lot():
    resultats, erreurs, nom_fichier = None, [], None
    if request.method == "POST":
        fichier = request.files.get("fichier")
        if not fichier or not fichier.filename:
            erreurs = ["Choisissez un fichier CSV."]
        elif not fichier.filename.lower().endswith((".csv", ".txt")):
            erreurs = ["Le fichier doit être au format CSV (.csv)."]
        else:
            nom_fichier = fichier.filename
            lignes, erreurs = lire_csv_lot(fichier.read())
            if lignes:
                probas = RESEAU.predire_proba([[l["valeurs"][c] for c in COLONNES] for l in lignes])
                resultats = []
                for l, p in zip(lignes, probas):
                    libelle, classe = niveau(float(p))
                    resultats.append({**l, "probabilite": round(float(p) * 100, 1),
                                      "verdict": "Réussite" if p > 0.5 else "Échec",
                                      "niveau": libelle, "classe": classe})
            elif not erreurs:
                erreurs = ["Aucune ligne valide dans le fichier."]
    resume = None
    if resultats:
        n_reussite = sum(r["verdict"] == "Réussite" for r in resultats)
        resume = {"total": len(resultats), "reussite": n_reussite, "echec": len(resultats) - n_reussite,
                  "moyenne": round(sum(r["probabilite"] for r in resultats) / len(resultats), 1)}
    return render_template("lot.html", resultats=resultats, erreurs=erreurs, resume=resume,
                           nom_fichier=nom_fichier, colonnes=COLONNES, page="lot")


@app.route("/exemple.csv")
def exemple_csv():
    return send_from_directory(os.path.join(DOSSIER, "exemples"), "nouveaux_etudiants.csv",
                               as_attachment=True, mimetype="text/csv")


@app.route("/modele")
def modele():
    L, H = 600, 240
    y_max = max(RESEAU.historique_train[0], RESEAU.historique_test[0]) * 1.1
    courbes = {
        "train": points_courbe(RESEAU.historique_train, L, H, y_max),
        "test": points_courbe(RESEAU.historique_test, L, H, y_max),
        "largeur": L, "hauteur": H,
        "graduations_y": [{"y": H - v / y_max * H, "texte": f"{v:.1f}".replace(".", ",")} for v in np.arange(0, y_max, 0.1)],
        "graduations_x": [{"x": e / RESEAU.n_epoques * L, "texte": str(e)} for e in range(0, RESEAU.n_epoques + 1, 500)],
    }
    architecture = {"entrees": 4, "caches": RESEAU.n_caches, "sorties": 1, "parametres": RESEAU.n_parametres,
                    "taux": RESEAU.taux, "epoques": RESEAU.n_epoques}
    # Positions des neurones pour le schéma SVG (x fixe par couche, y centrés)
    couches = []
    for x, n, ecart in [(130, 4, 58), (270, RESEAU.n_caches, 31), (400, 1, 0)]:
        couches.append([(x, 140 - (n - 1) * ecart / 2 + j * ecart) for j in range(n)])
    liens = [(a[0], a[1], b[0], b[1]) for i in range(2) for a in couches[i] for b in couches[i + 1]]
    schema = {"couches": couches, "liens": liens, "noms": ["étude", "sommeil", "présence", "note"]}
    return render_template("modele.html", metriques=METRIQUES, courbes=courbes,
                           architecture=architecture, schema=schema, page="modele")


@app.route("/donnees")
def donnees():
    stats = []
    for i, c in enumerate(COLONNES):
        col = X_DONNEES[:, i]
        stats.append({"colonne": c, **CARACTERISTIQUES[c], "min": col.min(), "max": col.max(),
                      "moyenne": col.mean(), "reussite": col[Y_DONNEES.ravel() == 1].mean(),
                      "echec": col[Y_DONNEES.ravel() == 0].mean()})
    # Nuage de points : heures d'étude (x, 0 à 20) / heures de sommeil (y, 4 à 11)
    L, H = 520, 320
    points = [{"x": X_DONNEES[i, 0] / 20 * L, "y": H - (X_DONNEES[i, 1] - 4) / 7 * H, "r": int(Y_DONNEES[i, 0])}
              for i in range(len(X_DONNEES))]
    lignes = [{**{c: X_DONNEES[i, j] for j, c in enumerate(COLONNES)}, "reussite": int(Y_DONNEES[i, 0])}
              for i in range(len(X_DONNEES))]
    n_reussite = int(Y_DONNEES.sum())
    return render_template("donnees.html", stats=stats, points=points, lignes=lignes, L=L, H=H,
                           n_total=len(lignes), n_reussite=n_reussite, colonnes=COLONNES, page="donnees")


@app.errorhandler(404)
def page_introuvable(e):
    return render_template("erreur.html", code=404, message="Cette page n'existe pas.", page=""), 404


@app.errorhandler(413)
def fichier_trop_gros(e):
    return render_template("lot.html", resultats=None, resume=None, nom_fichier=None, colonnes=COLONNES,
                           erreurs=["Le fichier dépasse 1 Mo."], page="lot"), 413


if __name__ == "__main__":
    print(f"Modèle entraîné : précision sur le test = {METRIQUES['precision']:.1%}")
    print("Ouvrez http://127.0.0.1:5000 dans votre navigateur.")
    app.run(debug=False)
