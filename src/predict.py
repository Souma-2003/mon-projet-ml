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
"""
predict.py — Script d'inférence (prédiction sur de nouveaux clients).

Ce script montre comment utiliser le modèle entraîné pour
prédire si un nouveau client va churner ou non.
"""

import json

import joblib
import numpy as np
import pandas as pd


def load_model_and_scaler():
    """Charge le modèle et le scaler sauvegardés."""
    model = joblib.load("models/model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler


def predict_churn(client_data: dict) -> dict:
    """
    Prédit si un client va churner.

    Args:
        client_data: dictionnaire avec les caractéristiques du client

    Returns:
        dict avec la prédiction (0/1) et la probabilité de churn
    """
    model, scaler = load_model_and_scaler()

    # Créer un DataFrame avec les données du client
    df = pd.DataFrame([client_data])

    # Appliquer la même normalisation qu'à l'entraînement
    df_scaled = scaler.transform(df)

    # Prédiction
    prediction = model.predict(df_scaled)[0]
    probabilities = model.predict_proba(df_scaled)[0]
    churn_probability = probabilities[1]  # Probabilité d'être churner

    result = {
        "prediction": int(prediction),
        "label": "CHURN" if prediction == 1 else "RESTE",
        "probabilite_churn": round(float(churn_probability), 4),
        "probabilite_reste": round(float(probabilities[0]), 4),
        "risque": (
            "ÉLEVÉ" if churn_probability > 0.7
            else "MOYEN" if churn_probability > 0.4
            else "FAIBLE"
        )
    }
    return result


# ─────────────────────────────────────────────────────────────
# Exemples de clients pour tester le modèle
# ─────────────────────────────────────────────────────────────
CLIENTS_EXEMPLES = [
    {
        "nom": "Client à haut risque de churn",
        "data": {
            "age": 28,
            "anciennete_mois": 3,        # Très récent
            "type_contrat": 0,           # 0 = Mensuel (plus de risque)
            "facture_mensuelle": 110.0,  # Facture élevée
            "facture_totale": 330.0,
            "telephone_multiple": 0,
            "internet": 2,               # 2 = Fibre (plus cher)
            "securite_en_ligne": 0,
            "sauvegarde_en_ligne": 0,
            "protection_appareil": 0,
            "support_tech": 0,           # Pas de support tech
            "streaming_tv": 1,
            "streaming_films": 1,
            "facture_electronique": 0,
            "mode_paiement": 3,          # 3 = Chèque papier
        }
    },
    {
        "nom": "Client fidèle (faible risque)",
        "data": {
            "age": 55,
            "anciennete_mois": 60,       # Client fidèle depuis 5 ans
            "type_contrat": 2,           # 2 = Biennal (moins de risque)
            "facture_mensuelle": 45.0,   # Facture raisonnable
            "facture_totale": 2700.0,
            "telephone_multiple": 1,
            "internet": 1,               # 1 = DSL
            "securite_en_ligne": 1,
            "sauvegarde_en_ligne": 1,
            "protection_appareil": 1,
            "support_tech": 1,           # A le support tech
            "streaming_tv": 0,
            "streaming_films": 0,
            "facture_electronique": 1,
            "mode_paiement": 0,          # 0 = Virement
        }
    },
]


def main():
    print("=" * 55)
    print("  PRÉDICTIONS DE CHURN - MODÈLE TÉLÉCOM")
    print("=" * 55)

    # Charger les métriques du modèle pour afficher le contexte
    with open("models/metrics.json", "r") as f:
        metrics = json.load(f)
    print(f"Modèle chargé | ROC-AUC : {metrics['roc_auc']} | F1 : {metrics['f1_score']}\n")

    for exemple in CLIENTS_EXEMPLES:
        print(f"Client : {exemple['nom']}")
        print("-" * 40)
        result = predict_churn(exemple["data"])
        print(f"  Prédiction       : {result['label']}")
        print(f"  Probabilité churn: {result['probabilite_churn']:.2%}")
        print(f"  Niveau de risque : {result['risque']}")
        print()


if __name__ == "__main__":
    main()