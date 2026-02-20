"""
Fait des prédictions sur de nouveaux clients.
"""

import pandas as pd
import numpy as np
import pickle
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_model(filepath):
    """Charger le modèle entraîné."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def predict_churn(model, customer_data):
    """
    Prédire si un client va churner.
    
    Args:
        model: Modèle RandomForest entraîné
        customer_data: DataFrame avec les features du client
        
    Returns:
        tuple: (predictions, probabilities)
    """
    feature_cols = [
        'age', 'anciennete_mois', 'type_contrat', 'facture_mensuelle',
        'facture_totale', 'telephone_multiple', 'internet', 'securite_en_ligne',
        'sauvegarde_en_ligne', 'protection_appareil', 'support_tech',
        'streaming_tv', 'streaming_films', 'facture_electronique', 'mode_paiement'
    ]
    
    X = customer_data[feature_cols]
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    
    return predictions, probabilities


if __name__ == '__main__':
    # Charger le modèle
    model = load_model('models/model.pkl')
    logger.info("Modèle chargé avec succès")
    
    # Exemple 1: Client avec ancienneté élevée
    client_1 = {
        'age': 45,
        'anciennete_mois': 60,
        'type_contrat': 2,
        'facture_mensuelle': 85.5,
        'facture_totale': 5130,
        'telephone_multiple': 1,
        'internet': 2,
        'securite_en_ligne': 1,
        'sauvegarde_en_ligne': 1,
        'protection_appareil': 1,
        'support_tech': 1,
        'streaming_tv': 1,
        'streaming_films': 1,
        'facture_electronique': 1,
        'mode_paiement': 2
    }
    
    # Exemple 2: Client avec ancienneté faible
    client_2 = {
        'age': 28,
        'anciennete_mois': 2,
        'type_contrat': 0,
        'facture_mensuelle': 65.0,
        'facture_totale': 130,
        'telephone_multiple': 0,
        'internet': 1,
        'securite_en_ligne': 0,
        'sauvegarde_en_ligne': 0,
        'protection_appareil': 0,
        'support_tech': 0,
        'streaming_tv': 0,
        'streaming_films': 0,
        'facture_electronique': 0,
        'mode_paiement': 0
    }
    
    # Faire les prédictions
    clients_df = pd.DataFrame([client_1, client_2])
    predictions, probabilities = predict_churn(model, clients_df)
    
    logger.info("\n🔮 Prédictions de churn:")
    logger.info(f"Client 1: Churn={predictions[0]} | Probabilité={probabilities[0]:.2%}")
    logger.info(f"Client 2: Churn={predictions[1]} | Probabilité={probabilities[1]:.2%}")
