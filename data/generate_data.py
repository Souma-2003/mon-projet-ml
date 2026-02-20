"""
Génère 5000 clients réalistes pour l'entraînement du modèle.
"""

import pandas as pd
import numpy as np
from faker import Faker
import random

def generate_clients_data(n_clients=5000, seed=42):
    """
    Génère des données réalistes de clients.
    
    Args:
        n_clients: Nombre de clients à générer
        seed: Graine pour la reproductibilité
        
    Returns:
        pd.DataFrame: Dataset généré
    """
    np.random.seed(seed)
    random.seed(seed)
    fake = Faker('fr_FR')
    
    data = {
        'client_id': range(1, n_clients + 1),
        'nom': [fake.name() for _ in range(n_clients)],
        'email': [fake.email() for _ in range(n_clients)],
        'age': np.random.randint(18, 80, n_clients),
        'anciennete_mois': np.random.randint(0, 120, n_clients),
        'type_contrat': np.random.randint(0, 3, n_clients),
        'facture_mensuelle': np.random.uniform(20, 150, n_clients),
        'facture_totale': np.random.uniform(100, 9000, n_clients),
        'telephone_multiple': np.random.randint(0, 2, n_clients),
        'internet': np.random.randint(0, 3, n_clients),
        'securite_en_ligne': np.random.randint(0, 2, n_clients),
        'sauvegarde_en_ligne': np.random.randint(0, 2, n_clients),
        'protection_appareil': np.random.randint(0, 2, n_clients),
        'support_tech': np.random.randint(0, 2, n_clients),
        'streaming_tv': np.random.randint(0, 2, n_clients),
        'streaming_films': np.random.randint(0, 2, n_clients),
        'facture_electronique': np.random.randint(0, 2, n_clients),
        'mode_paiement': np.random.randint(0, 4, n_clients),
        'target': np.random.choice([0, 1], n_clients, p=[0.6, 0.4]),
    }
    
    df = pd.DataFrame(data)
    return df


def save_data(df, filepath='data/data.csv'):
    """Enregistrer le dataset en CSV."""
    df.to_csv(filepath, index=False)
    print(f"✓ Dataset sauvegardé: {filepath}")
    print(f"  Forme: {df.shape}")
    print(f"  Taux de churn: {df['target'].mean():.2%}")


if __name__ == '__main__':
    clients_df = generate_clients_data(n_clients=5000)
    save_data(clients_df)
