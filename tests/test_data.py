"""
test_data.py — Tests automatiques sur la qualité des données.

Ces tests sont lancés par GitHub Actions à chaque push.
Si un test échoue, le pipeline s'arrête et le modèle n'est pas entraîné.
"""

import os

import numpy as np
import pandas as pd
import pytest

# ─────────────────────────────────────────────────────────────
# FIXTURES : préparer les objets réutilisables dans les tests
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def df():
    """Charger le dataset une seule fois pour tous les tests."""
    return pd.read_csv("data/data.csv")


# ─────────────────────────────────────────────────────────────
# TESTS D'EXISTENCE ET DE STRUCTURE
# ─────────────────────────────────────────────────────────────

def test_fichier_csv_existe():
    """Le fichier de données doit exister."""
    assert os.path.exists("data/data.csv"), \
        "ERREUR : data/data.csv est introuvable ! Avez-vous généré les données ?"


def test_dataset_non_vide(df):
    """Le dataset ne doit pas être vide."""
    assert len(df) > 0, "Le dataset est vide !"


def test_nombre_minimum_de_lignes(df):
    """On a besoin d'au moins 500 exemples pour entraîner un modèle fiable."""
    assert len(df) >= 500, \
        f"Pas assez de données : {len(df)} lignes (minimum : 500)"


def test_colonnes_attendues_presentes(df):
    """Toutes les colonnes attendues doivent être présentes."""
    colonnes_attendues = [
        "age", "anciennete_mois", "type_contrat", "facture_mensuelle",
        "facture_totale", "telephone_multiple", "internet", "securite_en_ligne",
        "sauvegarde_en_ligne", "protection_appareil", "support_tech",
        "streaming_tv", "streaming_films", "facture_electronique",
        "mode_paiement", "target"
    ]
    for col in colonnes_attendues:
        assert col in df.columns, \
            f"Colonne manquante : '{col}'"


def test_colonne_target_existe(df):
    """La colonne cible (target) doit exister."""
    assert "target" in df.columns, "La colonne 'target' est manquante !"


# ─────────────────────────────────────────────────────────────
# TESTS DE QUALITÉ DES DONNÉES
# ─────────────────────────────────────────────────────────────

def test_pas_de_valeurs_nulles(df):
    """Aucune valeur manquante n'est acceptable."""
    nulls = df.isnull().sum()
    colonnes_avec_nulls = nulls[nulls > 0]
    assert len(colonnes_avec_nulls) == 0, \
        f"Valeurs manquantes détectées :\n{colonnes_avec_nulls}"


def test_target_est_binaire(df):
    """La variable cible doit être 0 ou 1 uniquement."""
    valeurs_uniques = set(df["target"].unique())
    assert valeurs_uniques.issubset({0, 1}), \
        f"La colonne 'target' contient des valeurs inattendues : {valeurs_uniques}"


def test_target_a_les_deux_classes(df):
    """
    Les deux classes (0 et 1) doivent être présentes.
    Un modèle entraîné sur une seule classe ne sert à rien.
    """
    assert df["target"].nunique() == 2, \
        "La colonne 'target' ne contient qu'une seule classe !"


def test_taux_de_churn_raisonnable(df):
    """
    Le taux de churn doit être entre 5% et 60%.
    Un taux extrême suggère un problème dans les données.
    """
    taux_churn = df["target"].mean()
    assert 0.05 <= taux_churn <= 0.60, \
        f"Taux de churn anormal : {taux_churn:.2%} (attendu entre 5% et 60%)"


def test_age_dans_les_limites(df):
    """L'âge des clients doit être réaliste (18-100 ans)."""
    assert df["age"].min() >= 18, \
        f"Âge minimum anormal : {df['age'].min()} (minimum attendu : 18)"
    assert df["age"].max() <= 100, \
        f"Âge maximum anormal : {df['age'].max()} (maximum attendu : 100)"


def test_anciennete_positive(df):
    """L'ancienneté en mois doit être positive ou nulle."""
    assert (df["anciennete_mois"] >= 0).all(), \
        "Des valeurs d'ancienneté négatives ont été détectées !"


def test_facture_mensuelle_positive(df):
    """La facture mensuelle doit être positive."""
    assert (df["facture_mensuelle"] > 0).all(), \
        "Des factures mensuelles négatives ou nulles ont été détectées !"


def test_colonnes_binaires_valides(df):
    """
    Les colonnes binaires (0/1) ne doivent contenir que 0 ou 1.
    """
    colonnes_binaires = [
        "telephone_multiple", "securite_en_ligne", "sauvegarde_en_ligne",
        "protection_appareil", "support_tech", "streaming_tv",
        "streaming_films", "facture_electronique"
    ]
    for col in colonnes_binaires:
        valeurs = set(df[col].unique())
        assert valeurs.issubset({0, 1}), \
            f"Colonne '{col}' contient des valeurs non binaires : {valeurs}"


def test_type_contrat_valide(df):
    """Le type de contrat doit être 0, 1 ou 2."""
    assert set(df["type_contrat"].unique()).issubset({0, 1, 2}), \
        "Type de contrat invalide détecté !"


def test_internet_valide(df):
    """Le type d'internet doit être 0 (Non), 1 (DSL) ou 2 (Fibre)."""
    assert set(df["internet"].unique()).issubset({0, 1, 2}), \
        "Valeur d'internet invalide détectée !"


def test_pas_de_doublons(df):
    """Il ne doit pas y avoir de lignes dupliquées."""
    nb_doublons = df.duplicated().sum()
    assert nb_doublons == 0, \
        f"{nb_doublons} lignes dupliquées détectées dans le dataset !"


def test_types_numeriques(df):
    """Toutes les colonnes (sauf target) doivent être numériques."""
    features = df.drop("target", axis=1)
    for col in features.columns:
        assert pd.api.types.is_numeric_dtype(features[col]), \
            f"La colonne '{col}' n'est pas numérique (type : {features[col].dtype})"