# 🤖 CI/CD Pipeline pour ML — Prédiction de Churn Télécom

Ce projet démontre comment mettre en place un **pipeline CI/CD complet**
pour un projet de Machine Learning en utilisant **GitHub Actions**.

## 🎯 Objectif du modèle

Prédire si un client télécom va **churner** (résilier son abonnement)
en se basant sur ses caractéristiques : type de contrat, ancienneté,
facture mensuelle, services souscrits, etc.

---

## 🏗️ Structure du projet

```
mon-projet-ml/
├── .github/
│   └── workflows/
│       ├── ci.yml          ← Pipeline CI (tests + entraînement)
│       └── cd.yml          ← Pipeline CD (déploiement production)
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
│   └── metrics.json        ← Métriques de performance
│
├── requirements.txt        ← Dépendances Python
└── README.md              ← Ce fichier
```

---

## ⚡ Lancer le projet en local

### 1. Cloner le repo et installer les dépendances

```bash
git clone https://github.com/TON_USERNAME/mon-projet-ml.git
cd mon-projet-ml
pip install -r requirements.txt
```

### 2. Générer le jeu de données

```bash
python data/generate_data.py
# → Crée data/data.csv avec 5000 clients fictifs
```

### 3. Entraîner le modèle

```bash
python src/train.py
# → Crée models/model.pkl
```

### 4. Évaluer les performances

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

## 📊 Résultats attendus

| Métrique | Seuil minimum |
|----------|---------------|
| Accuracy | 0.70          |
| Precision | 0.60         |
| Recall | 0.60           |
| F1-Score | 0.60          |
| ROC-AUC  | 0.75          |

---

## 🔄 Comment fonctionne le Pipeline CI/CD

### CI Pipeline (`.github/workflows/ci.yml`)

Se déclenche à chaque **git push** sur `main` :

```
┌─────────────────────────────────────┐
│ 1. Tests qualité des données (15)   │
│    ✓ Fichier existe                 │
│    ✓ Pas de valeurs nulles          │
│    ✓ Target binaire                 │
│    ...                              │
└──────────────┬──────────────────────┘
               │ Si ✅ seulement
               ▼
┌─────────────────────────────────────┐
│ 2. Entraînement du modèle           │
│    ✓ Génération des données         │
│    ✓ Entraînement RandomForest      │
│    ✓ Évaluation des métriques       │
│    ✓ Tests du modèle (14)           │
└──────────────┬──────────────────────┘
               │ Si ✅ seulement
               ▼
┌─────────────────────────────────────┐
│ 3. Upload des artifacts             │
│    ✓ model.pkl                      │
│    ✓ metrics.json                   │
└─────────────────────────────────────┘
```

### CD Pipeline (`.github/workflows/cd.yml`)

Se déclenche quand le CI réussit :

```
┌─────────────────────────────────────┐
│ CI Pipeline réussi                  │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 1. Entraînement en production       │
│    ✓ Génération des données         │
│    ✓ Entraînement du modèle         │
│    ✓ Validation performance         │
└──────────────┬──────────────────────┘
               │ Si ✅ seulement
               ▼
┌─────────────────────────────────────┐
│ 2. Upload artifacts finaux          │
│    ✓ Model entraîné                 │
│    ✓ Métriques                      │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 3. Prédictions de test              │
│    ✓ 2 clients exemples             │
└─────────────────────────────────────┘
```

---

## 🚀 Voir le pipeline en action sur GitHub

### Prérequis

- Un compte GitHub (gratuit sur [github.com](https://github.com))
- Git installé sur votre machine
- Un Personal Access Token (PAT)

### Étapes

1. **Créer un repo sur GitHub**
   - Allez sur https://github.com/new
   - Nom: `mon-projet-ml`
   - Cliquez `Create repository`

2. **Configurer git localement**
   ```bash
   git config user.name "Votre Nom"
   git config user.email "votre@email.com"
   ```

3. **Pusher le code sur GitHub**
   ```bash
   git branch -M main
   git remote add origin https://github.com/VOTRE_USERNAME/mon-projet-ml.git
   git push -u origin main
   ```

4. **Voir le pipeline s'exécuter**
   - Allez sur https://github.com/VOTRE_USERNAME/mon-projet-ml
   - Cliquez sur l'onglet **Actions**
   - Vous verrez les workflows CI/CD s'exécuter en temps réel! 🎉

### Statuts possibles

- 🟢 **Success** — Tous les tests passent
- 🔴 **Failure** — Un test a échoué
- 🟡 **In Progress** — Le workflow s'exécute actuellement

---

## 📝 Tests inclus

### 15 tests de qualité des données
1. Fichier CSV existe
2. Dataset non vide
3. Minimum 500 lignes
4. Colonnes attendues présentes
5. Colonne 'target' existe
6. Pas de valeurs nulles
7. Target est binaire (0/1)
8. Deux classes présentes
9. Taux de churn raisonnable (5-60%)
10. Âge dans les limites (18-100)
11. Ancienneté positive
12. Facture mensuelle positive
13. Colonnes binaires valides
14. Pas de doublons
15. Types numériques corrects

### 14 tests de validation du modèle
1. Modèle est un classificateur
2. Prédictions ont la bonne forme
3. Probabilités ont la bonne forme
4. Probabilités somment à 1
5. Accuracy positive
6. Modèle sérialisable
7. Modèle désérialisable
8. Feature importance disponible
9. Prédictions cohérentes
10. Validation d'entrée
11. Bon nombre d'estimateurs
12. Prédictions en plage valide
13. Classes trackées
14. Profondeur des arbres raisonnable

---

## 🛠️ Technologies utilisées

- **Python 3.10** — Langage principal
- **scikit-learn** — Modèle RandomForest
- **pandas / numpy** — Traitement des données
- **pytest** — Tests automatiques
- **faker** — Génération de données réalistes
- **GitHub Actions** — CI/CD

---

## 📚 Ressources utiles

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [scikit-learn RandomForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [pytest Documentation](https://docs.pytest.org/)

---

## 🎓 Apprentissages clés

Ce projet vous montre comment :
- ✅ Créer un pipeline ML automatisé
- ✅ Valider la qualité des données
- ✅ Entraîner et évaluer des modèles
- ✅ Utiliser GitHub Actions pour CI/CD
- ✅ Implémenter des tests automatiques
- ✅ Déployer en production de manière fiable

---

**Prêt à lancer votre premier pipeline CI/CD? 🚀**
