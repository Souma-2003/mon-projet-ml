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
