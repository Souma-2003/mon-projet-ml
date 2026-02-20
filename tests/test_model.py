"""
test_model.py — Tests automatiques sur le modèle entraîné.

Ces tests vérifient que le modèle :
1. A bien été sauvegardé
2. Peut faire des prédictions
3. Produit des prédictions valides et cohérentes
4. A de bonnes performances (métriques)
"""

import json
import os

import joblib
import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def model():
    """Charger le modèle entraîné."""
    assert os.path.exists("models/model.pkl"), \
        "ERREUR : models/model.pkl introuvable. Avez-vous lancé train.py ?"
    return joblib.load("models/model.pkl")


@pytest.fixture(scope="module")
def scaler():
    """Charger le scaler."""
    assert os.path.exists("models/scaler.pkl"), \
        "ERREUR : models/scaler.pkl introuvable."
    return joblib.load("models/scaler.pkl")


@pytest.fixture(scope="module")
def metrics():
    """Charger les métriques sauvegardées."""
    assert os.path.exists("models/metrics.json"), \
        "ERREUR : models/metrics.json introuvable."
    with open("models/metrics.json", "r") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def exemple_client():
    """Un exemple de client pour tester les prédictions."""
    return np.array([[
        35,       # age
        12,       # anciennete_mois
        0,        # type_contrat (Mensuel)
        75.0,     # facture_mensuelle
        900.0,    # facture_totale
        1,        # telephone_multiple
        1,        # internet (DSL)
        0,        # securite_en_ligne
        1,        # sauvegarde_en_ligne
        0,        # protection_appareil
        1,        # support_tech
        0,        # streaming_tv
        1,        # streaming_films
        1,        # facture_electronique
        0         # mode_paiement (Virement)
    ]])


# ─────────────────────────────────────────────────────────────
# TESTS D'EXISTENCE DES FICHIERS
# ─────────────────────────────────────────────────────────────

def test_fichier_modele_existe():
    """Le fichier modèle doit exister après l'entraînement."""
    assert os.path.exists("models/model.pkl"), \
        "Le modèle n'a pas été sauvegardé !"


def test_fichier_scaler_existe():
    """Le scaler doit exister après l'entraînement."""
    assert os.path.exists("models/scaler.pkl"), \
        "Le scaler n'a pas été sauvegardé !"


def test_fichier_metriques_existe():
    """Le fichier de métriques doit exister."""
    assert os.path.exists("models/metrics.json"), \
        "Le fichier metrics.json n'a pas été créé !"


# ─────────────────────────────────────────────────────────────
# TESTS DE PRÉDICTION
# ─────────────────────────────────────────────────────────────

def test_modele_peut_predire(model, scaler, exemple_client):
    """Le modèle doit pouvoir faire une prédiction sans erreur."""
    client_scaled = scaler.transform(exemple_client)
    prediction = model.predict(client_scaled)
    assert prediction is not None, "La prédiction est None !"
    assert len(prediction) == 1, "Une seule prédiction attendue !"


def test_prediction_est_binaire(model, scaler, exemple_client):
    """La prédiction doit être 0 (reste) ou 1 (churn)."""
    client_scaled = scaler.transform(exemple_client)
    prediction = model.predict(client_scaled)
    assert prediction[0] in [0, 1], \
        f"Prédiction invalide : {prediction[0]} (attendu : 0 ou 1)"


def test_probabilites_sommees_a_1(model, scaler, exemple_client):
    """Les probabilités pour chaque classe doivent sommer à 1."""
    client_scaled = scaler.transform(exemple_client)
    probas = model.predict_proba(client_scaled)[0]
    assert abs(sum(probas) - 1.0) < 1e-6, \
        f"Les probabilités ne somment pas à 1 : {sum(probas)}"


def test_probabilites_entre_0_et_1(model, scaler, exemple_client):
    """Chaque probabilité doit être entre 0 et 1."""
    client_scaled = scaler.transform(exemple_client)
    probas = model.predict_proba(client_scaled)[0]
    for p in probas:
        assert 0.0 <= p <= 1.0, \
            f"Probabilité invalide : {p} (doit être entre 0 et 1)"


def test_nombre_classes(model):
    """Le modèle doit avoir exactement 2 classes (0 et 1)."""
    assert len(model.classes_) == 2, \
        f"Nombre de classes inattendu : {len(model.classes_)} (attendu : 2)"
    assert list(model.classes_) == [0, 1], \
        f"Classes inattendues : {model.classes_} (attendu : [0, 1])"


def test_prediction_coherente_client_risque(model, scaler):
    """
    Un client à très haut risque (contrat mensuel, peu d'ancienneté,
    facture élevée, pas de support) doit avoir une probabilité de churn > 50%.
    """
    client_risque = np.array([[
        25, 2, 0, 115.0, 230.0, 0, 2, 0, 0, 0, 0, 1, 1, 0, 3
    ]])
    client_scaled = scaler.transform(client_risque)
    prob_churn = model.predict_proba(client_scaled)[0][1]
    assert prob_churn > 0.5, \
        f"Client à haut risque : probabilité de churn trop faible ({prob_churn:.2%})"


def test_prediction_coherente_client_fidele(model, scaler):
    """
    Un client fidèle (contrat biennal, longue ancienneté, facture basse,
    support tech) doit avoir une probabilité de churn < 30%.
    """
    client_fidele = np.array([[
        55, 60, 2, 40.0, 2400.0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0
    ]])
    client_scaled = scaler.transform(client_fidele)
    prob_churn = model.predict_proba(client_scaled)[0][1]
    assert prob_churn < 0.30, \
        f"Client fidèle : probabilité de churn trop élevée ({prob_churn:.2%})"


def test_predictions_sur_plusieurs_clients(model, scaler):
    """Le modèle doit gérer un batch de plusieurs clients à la fois."""
    clients = np.random.rand(10, 15)  # 10 clients, 15 features
    clients_scaled = scaler.transform(clients)
    predictions = model.predict(clients_scaled)
    assert len(predictions) == 10, \
        "Le modèle doit retourner 10 prédictions pour 10 clients"


# ─────────────────────────────────────────────────────────────
# TESTS DE PERFORMANCE (métriques)
# ─────────────────────────────────────────────────────────────

def test_accuracy_suffisante(metrics):
    """L'accuracy doit dépasser 75%."""
    assert metrics["accuracy"] >= 0.75, \
        f"Accuracy insuffisante : {metrics['accuracy']:.4f} (minimum : 0.75)"


def test_f1_score_suffisant(metrics):
    """Le F1-score doit dépasser 0.60."""
    assert metrics["f1_score"] >= 0.60, \
        f"F1-score insuffisant : {metrics['f1_score']:.4f} (minimum : 0.60)"


def test_roc_auc_suffisant(metrics):
    """Le ROC-AUC doit dépasser 0.75."""
    assert metrics["roc_auc"] >= 0.75, \
        f"ROC-AUC insuffisant : {metrics['roc_auc']:.4f} (minimum : 0.75)"


def test_metriques_json_contient_champs_requis(metrics):
    """Le fichier metrics.json doit contenir les champs requis."""
    champs_requis = ["accuracy", "f1_score", "roc_auc"]
    for champ in champs_requis:
        assert champ in metrics, \
            f"Champ manquant dans metrics.json : '{champ}'"