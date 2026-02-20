"""
Évalue le modèle et valide les seuils de performance.
"""

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report
)
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


PERFORMANCE_THRESHOLDS = {
    'accuracy': 0.70,
    'precision': 0.60,
    'recall': 0.60,
    'f1': 0.60,
    'roc_auc': 0.75
}


def load_model(filepath):
    """Charger le modèle."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def preprocess_data(df):
    """Préparer les données pour l'évaluation."""
    df = df.copy()
    
    feature_cols = [
        'age', 'anciennete_mois', 'type_contrat', 'facture_mensuelle',
        'facture_totale', 'telephone_multiple', 'internet', 'securite_en_ligne',
        'sauvegarde_en_ligne', 'protection_appareil', 'support_tech',
        'streaming_tv', 'streaming_films', 'facture_electronique', 'mode_paiement'
    ]
    
    X = df[feature_cols]
    y = df['target']
    
    return X, y


def evaluate_model(model, X_test, y_test):
    """
    Évaluer le modèle.
    Retourne les métriques et valide les seuils.
    """
    logger.info("Évaluation du modèle...")
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }
    
    # Afficher les résultats
    logger.info("\n📊 Métriques de performance:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Valider les seuils
    logger.info("\n✓ Validation des seuils:")
    all_passed = True
    for metric, threshold in PERFORMANCE_THRESHOLDS.items():
        passed = metrics[metric] >= threshold
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"  {metric}: {metrics[metric]:.4f} >= {threshold} {status}")
        if not passed:
            all_passed = False
    
    # Rapport détaillé
    logger.info("\n📋 Rapport de classification:")
    logger.info(classification_report(y_test, y_pred))
    
    return metrics, all_passed


if __name__ == '__main__':
    # Charger les données
    df = pd.read_csv('data/data.csv')
    X, y = preprocess_data(df)
    
    # Diviser les données
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Charger et évaluer le modèle
    model = load_model('models/model.pkl')
    metrics, passed = evaluate_model(model, X_test, y_test)
    
    # Sauvegarder les métriques
    with open('models/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    if passed:
        logger.info("\n✓ Tous les seuils de performance sont atteints!")
    else:
        logger.warning("\n✗ Certains seuils ne sont pas atteints!")
