# -*- coding: utf-8 -*-
"""
Tests automatiques de l'application.
Lancement :  python -m pytest -q
"""
import io

import pytest

from app import METRIQUES, app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_modele_bien_entraine():
    # Le modèle doit atteindre au moins 85 % sur le test (96,7 % attendu)
    assert METRIQUES["precision"] >= 0.85


@pytest.mark.parametrize("url", ["/", "/lot", "/modele", "/donnees", "/exemple.csv"])
def test_pages_accessibles(client, url):
    assert client.get(url).status_code == 200


def test_page_inconnue(client):
    assert client.get("/nexiste-pas").status_code == 404


def test_api_prediction(client):
    rep = client.post("/api/predire", json={"heures_etude": 15, "heures_sommeil": 7.5,
                                            "taux_presence": 90, "note_precedente": 14})
    assert rep.status_code == 200
    d = rep.get_json()
    assert 0 <= d["probabilite"] <= 100
    assert d["verdict"] == "Réussite"


def test_api_effet_du_sommeil(client):
    base = {"heures_etude": 15, "taux_presence": 90, "note_precedente": 14}
    bien = client.post("/api/predire", json={**base, "heures_sommeil": 7.5}).get_json()["probabilite"]
    peu = client.post("/api/predire", json={**base, "heures_sommeil": 4.5}).get_json()["probabilite"]
    assert bien > peu


def test_api_virgule_decimale(client):
    rep = client.post("/api/predire", json={"heures_etude": "12,5", "heures_sommeil": "7,5",
                                            "taux_presence": "80", "note_precedente": "12"})
    assert rep.status_code == 200


def test_api_valeurs_invalides(client):
    rep = client.post("/api/predire", json={"heures_etude": "abc", "heures_sommeil": 30,
                                            "taux_presence": 90, "note_precedente": 14})
    assert rep.status_code == 400
    assert len(rep.get_json()["erreurs"]) == 2


def test_api_sans_json(client):
    assert client.post("/api/predire", data="pas du json").status_code == 400


def test_formulaire_sans_javascript(client):
    rep = client.post("/", data={"heures_etude": "2", "heures_sommeil": "8",
                                 "taux_presence": "50", "note_precedente": "7"})
    assert rep.status_code == 200
    assert "Échec".encode() in rep.data


def test_lot_fichier_exemple(client):
    with open("exemples/nouveaux_etudiants.csv", "rb") as f:
        rep = client.post("/lot", data={"fichier": (f, "etudiants.csv")}, content_type="multipart/form-data")
    assert rep.status_code == 200
    assert b"Amina" in rep.data and "Télécharger les résultats".encode() in rep.data


def test_lot_point_virgule_et_erreurs(client):
    contenu = "nom;heures_etude;heures_sommeil;taux_presence;note_precedente\nA;10;7,5;80;12\nB;x;7;80;12\n"
    rep = client.post("/lot", data={"fichier": (io.BytesIO(contenu.encode()), "lot.csv")},
                      content_type="multipart/form-data")
    assert rep.status_code == 200
    assert "Ligne 3 ignorée".encode() in rep.data


def test_lot_colonnes_manquantes(client):
    rep = client.post("/lot", data={"fichier": (io.BytesIO(b"a,b\n1,2\n"), "lot.csv")},
                      content_type="multipart/form-data")
    assert "Colonnes manquantes".encode() in rep.data


def test_lot_mauvaise_extension(client):
    rep = client.post("/lot", data={"fichier": (io.BytesIO(b"x"), "image.png")},
                      content_type="multipart/form-data")
    assert "format CSV".encode() in rep.data
