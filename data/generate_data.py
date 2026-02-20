"""
Script pour générer un jeu de données réaliste de Churn Télécom.
Ce script crée un fichier data.csv avec 5000 clients fictifs.
"""
import numpy as np
import pandas as pd

np.random.seed(42)
n = 5000

age = np.random.randint(18, 75, n)
anciennete_mois = np.random.randint(1, 72, n)
type_contrat = np.random.choice(["Mensuel", "Annuel", "Biennal"], n, p=[0.5, 0.3, 0.2])
facture_mensuelle = np.round(np.random.uniform(20, 120, n), 2)
facture_totale = np.round(facture_mensuelle * anciennete_mois * np.random.uniform(0.9, 1.1, n), 2)
telephone_multiple = np.random.randint(0, 2, n)
internet = np.random.choice(["Non", "DSL", "Fibre"], n, p=[0.1, 0.4, 0.5])
securite_en_ligne = np.random.randint(0, 2, n)
sauvegarde_en_ligne = np.random.randint(0, 2, n)
protection_appareil = np.random.randint(0, 2, n)
support_tech = np.random.randint(0, 2, n)
streaming_tv = np.random.randint(0, 2, n)
streaming_films = np.random.randint(0, 2, n)
mode_paiement = np.random.choice(["Virement", "Carte bancaire", "Chèque électronique", "Chèque papier"], n, p=[0.3, 0.3, 0.25, 0.15])
facture_electronique = np.random.randint(0, 2, n)

# Signaux plus forts pour un meilleur modèle
prob_churn = np.zeros(n)
prob_churn += np.where(type_contrat == "Mensuel", 0.35, 0.0)
prob_churn += np.where(type_contrat == "Annuel", 0.08, 0.0)
prob_churn += np.where(type_contrat == "Biennal", 0.02, 0.0)
prob_churn += (facture_mensuelle - 20) / 100 * 0.30
prob_churn += np.where(anciennete_mois < 6, 0.25, 0.0)
prob_churn += np.where(anciennete_mois < 12, 0.15, 0.0)
prob_churn += np.where(anciennete_mois > 36, -0.15, 0.0)
prob_churn += np.where(support_tech == 0, 0.10, -0.10)
prob_churn += np.where(securite_en_ligne == 0, 0.05, -0.05)
prob_churn += np.where(internet == "Fibre", 0.10, 0.0)
prob_churn += np.where(mode_paiement == "Chèque papier", 0.10, 0.0)
prob_churn += np.where(facture_electronique == 0, 0.05, -0.02)

prob_churn = np.clip(prob_churn, 0.02, 0.95)
churn = np.random.binomial(1, prob_churn)

type_contrat_enc = pd.Categorical(type_contrat, categories=["Mensuel", "Annuel", "Biennal"]).codes
internet_enc = pd.Categorical(internet, categories=["Non", "DSL", "Fibre"]).codes
paiement_enc = pd.Categorical(mode_paiement, categories=["Virement", "Carte bancaire", "Chèque électronique", "Chèque papier"]).codes

df = pd.DataFrame({
    "age": age,
    "anciennete_mois": anciennete_mois,
    "type_contrat": type_contrat_enc,
    "facture_mensuelle": facture_mensuelle,
    "facture_totale": facture_totale,
    "telephone_multiple": telephone_multiple,
    "internet": internet_enc,
    "securite_en_ligne": securite_en_ligne,
    "sauvegarde_en_ligne": sauvegarde_en_ligne,
    "protection_appareil": protection_appareil,
    "support_tech": support_tech,
    "streaming_tv": streaming_tv,
    "streaming_films": streaming_films,
    "facture_electronique": facture_electronique,
    "mode_paiement": paiement_enc,
    "target": churn
})

df.to_csv("data/data.csv", index=False)
print(f"Dataset généré : {len(df)} clients")
print(f"Taux de churn : {churn.mean():.2%}")
print(f"Colonnes : {list(df.columns)}")
print(df.head())