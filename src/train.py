"""
Entraîne un modèle RandomForest pour prédire le churn.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_data(filepath):
    """Charger le dataset."""
    logger.info(f"Chargement des données depuis {filepath}")
    return pd.read_csv(filepath)


def preprocess_data(df):
    """Préparer les données pour l'entraînement."""
    logger.info("Prétraitement des données...")
    
    df = df.copy()
    
    # Sélectionner les features
    feature_cols = [
        'age', 'anciennete_mois', 'type_contrat', 'facture_mensuelle',
        'facture_totale', 'telephone_multiple', 'internet', 'securite_en_ligne',
        'sauvegarde_en_ligne', 'protection_appareil', 'support_tech',
        'streaming_tv', 'streaming_films', 'facture_electronique', 'mode_paiement'
    ]
    
    X = df[feature_cols]
    y = df['target']
    
    return X, y


def train_model(X_train, y_train):
    """Entraîner le modèle RandomForest."""
    logger.info("Entraînement du modèle RandomForest...")
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    logger.info("✓ Entraînement terminé")
    
    return model


def save_model(model, scaler, filepath='models/model.pkl'):
    """Enregistrer le modèle entraîné."""
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"✓ Modèle sauvegardé: {filepath}")


if __name__ == '__main__':
    # Charger et préparer les données
    df = load_data('data/data.csv')
    X, y = preprocess_data(df)
    
    # Diviser les données
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    logger.info(f"Ensemble d'entraînement: {X_train.shape[0]} échantillons")
    logger.info(f"Ensemble de test: {X_test.shape[0]} échantillons")
    
    # Entraîner le modèle
    model = train_model(X_train, y_train)
    
    # Évaluer
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    logger.info(f"Accuracy train: {train_score:.4f}")
    logger.info(f"Accuracy test: {test_score:.4f}")
    
    # Sauvegarder
    save_model(model, None)
"""
train.py — Entraînement du modèle de prédiction de Churn Télécom.

Ce script :
1. Charge le jeu de données
2. Prétraite les features
3. Entraîne un RandomForestClassifier
4. Sauvegarde le modèle + le scaler + les métriques
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(filepath: str) -> pd.DataFrame:
    """Charge le dataset depuis un fichier CSV."""
    if not os.path.exists(filepath):
        print(f"ERREUR : Fichier introuvable → {filepath}")
        sys.exit(1)
    df = pd.read_csv(filepath)
    print(f"Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df


def preprocess(df: pd.DataFrame):
    """
    Sépare les features (X) de la cible (y),
    puis divise en ensembles d'entraînement et de test.
    """
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
        # stratify=y : garder la même proportion de churn dans train et test
    )
    print(f"Train : {X_train.shape[0]} exemples | Test : {X_test.shape[0]} exemples")
    print(f"Taux de churn (train) : {y_train.mean():.2%}")
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """
    Normalise les features pour que toutes soient sur la même échelle.
    Le scaler est entraîné UNIQUEMENT sur X_train pour éviter le data leakage.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def train_model(X_train, y_train):
    """
    Entraîne un RandomForest avec des hyperparamètres optimisés.
    class_weight='balanced' est crucial quand les classes sont déséquilibrées
    (ici ~73% restent, ~27% churners).
    """
    model = RandomForestClassifier(
        n_estimators=300,        # Plus d'arbres = meilleure stabilité
        max_depth=12,            # Arbres plus profonds pour capturer la complexité
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",     # Nombre de features par arbre
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"  # Compense le déséquilibre de classes
    )
    model.fit(X_train, y_train)
    print("Modèle entraîné avec succès.")
    return model


def evaluate_model(model, X_test, y_test, feature_names):
    """
    Évalue le modèle et retourne les métriques importantes.
    Pour le churn, on s'intéresse surtout au F1-score et ROC-AUC
    car les classes sont souvent déséquilibrées.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probabilité d'être churner

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
    }

    print("\n===== RÉSULTATS D'ÉVALUATION =====")
    print(f"Accuracy  : {metrics['accuracy']:.4f}")
    print(f"F1-Score  : {metrics['f1_score']:.4f}")
    print(f"ROC-AUC   : {metrics['roc_auc']:.4f}")
    print("\nRapport de classification détaillé :")
    print(classification_report(y_test, y_pred, target_names=["Reste", "Churn"]))

    # Importance des features
    importances = model.feature_importances_
    feature_importance = sorted(
        zip(feature_names, importances), key=lambda x: x[1], reverse=True
    )
    print("\nTop 5 features les plus importantes :")
    for name, importance in feature_importance[:5]:
        print(f"  {name}: {importance:.4f}")

    metrics["feature_importance"] = {k: round(v, 4) for k, v in feature_importance}
    return metrics


def save_artifacts(model, scaler, metrics):
    """Sauvegarde le modèle, le scaler et les métriques dans le dossier models/."""
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nFichiers sauvegardés dans models/")
    print("  → models/model.pkl")
    print("  → models/scaler.pkl")
    print("  → models/metrics.json")


def main():
    print("=" * 50)
    print("  ENTRAÎNEMENT DU MODÈLE DE CHURN TÉLÉCOM")
    print("=" * 50)

    df = load_data("data/data.csv")
    X_train, X_test, y_train, y_test = preprocess(df)
    X_train_s, X_test_s, scaler = scale_features(X_train, X_test)
    model = train_model(X_train_s, y_train)
    metrics = evaluate_model(model, X_test_s, y_test, list(X_train.columns))
    save_artifacts(model, scaler, metrics)

    print("\nEntraînement terminé avec succès !")


if __name__ == "__main__":
    main()