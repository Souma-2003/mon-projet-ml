# 🤖 CI/CD Pipeline pour ML — Prédiction de Churn Télécom

Ce projet démontre comment mettre en place un pipeline CI/CD complet
pour un projet de Machine Learning en utilisant GitHub Actions.

## 🎯 Objectif du modèle

Prédire si un client télécom va **churner** (résilier son abonnement)
en se basant sur ses caractéristiques : type de contrat, ancienneté,
facture mensuelle, services souscrits, etc.

---

## 🏗️ Structure du projet

```
ml-churn-cicd/
├── .github/
│   └── workflows/
│       ├── ci.yml          ← Pipeline CI (qualité + tests + entraînement)
│       └── cd.yml          ← Pipeline CD (déploiement staging + production)
│
├── data/
│   ├── generate_data.py    ← Générateur du jeu de données
│   └── data.csv            ← Dataset généré (5000 clients)
│
├── src/
│   ├── train.py            ← Entraînement du modèle RandomForest
│   ├── evaluate.py         ← Validation des métriques (seuils min.)
│   └── predict.py          ← Inférence sur de nouveaux clients
│
├── tests/
│   ├── test_data.py        ← 15 tests de qualité des données
│   └── test_model.py       ← 14 tests de validation du modèle
│
├── models/                 ← Générés automatiquement
│   ├── model.pkl           ← Modèle RandomForest entraîné
│   ├── scaler.pkl          ← StandardScaler
│   └── metrics.json        ← Métriques de performance
│
└── requirements.txt
```

---

## ⚡ Lancer le projet en local

### 1. Cloner le repo et installer les dépendances
```bash
git clone https://github.com/TON_USERNAME/ml-churn-cicd.git
cd ml-churn-cicd
pip install -r requirements.txt
```

### 2. Générer le jeu de données
```bash
python data/generate_data.py
# → Crée data/data.csv avec 5000 clients fictifs (taux de churn ~40%)
```

### 3. Entraîner le modèle
```bash
python src/train.py
# → Crée models/model.pkl, models/scaler.pkl, models/metrics.json
```

### 4. Valider les performances
```bash
python src/evaluate.py
# → Vérifie que Accuracy ≥ 0.70, F1 ≥ 0.60, ROC-AUC ≥ 0.75
```

### 5. Lancer les tests
```bash
pytest tests/ -v
# → Lance les 29 tests automatiques
```

### 6. Tester les prédictions
```bash
python src/predict.py
# → Prédit le churn pour 2 clients exemples
```

---

## 📊 Résultats du modèle

| Métrique | Valeur | Seuil minimum |
|----------|--------|---------------|
| Accuracy | 0.713  | 0.70          |
| F1-Score | 0.641  | 0.60          |
| ROC-AUC  | 0.786  | 0.75          |

**Features les plus importantes :**
1. Ancienneté (mois)
2. Facture mensuelle
3. Facture totale
4. Type de contrat
5. Âge

---

## 🔄 Comment fonctionne le Pipeline CI/CD

### CI Pipeline (`.github/workflows/ci.yml`)

Se déclenche à chaque `git push` :

```
Push →  Job 1: Qualité du code (flake8, black)
              ↓ (si ✅)
        Job 2: Validation des données (15 tests pytest)
              ↓ (si ✅)
        Job 3: Entraînement + Évaluation + Tests modèle (14 tests)
```

### CD Pipeline (`.github/workflows/cd.yml`)

Se déclenche quand le CI réussit sur `main` :

```
CI réussi →  Job 1: Vérification du statut CI
                   ↓
             Job 2: Déploiement Staging
                   ↓ (si ✅)
             Job 3: Déploiement Production
```

---

## 🔐 Configurer les secrets GitHub

Pour les déploiements réels, ajouter dans `Settings → Secrets` :

```
AWS_ACCESS_KEY_ID      ← Pour déployer sur AWS
AWS_SECRET_ACCESS_KEY
DATABASE_URL           ← Pour logger les métriques
SLACK_WEBHOOK_URL      ← Pour les notifications
```

---

## 📈 Voir le pipeline en action

1. Fork ce repo sur GitHub
2. Active les GitHub Actions (onglet `Actions`)
3. Fais un `git push` sur `main`
4. Va dans l'onglet `Actions` → tu verras le pipeline s'exécuter en temps réel

---

## 🛠️ Technologies utilisées

- **Python 3.10** — Langage principal
- **scikit-learn** — Modèle RandomForest
- **pandas / numpy** — Traitement des données
- **pytest** — Tests automatiques
- **flake8 / black** — Qualité du code
- **GitHub Actions** — CI/CD
- **joblib** — Sérialisation du modèle