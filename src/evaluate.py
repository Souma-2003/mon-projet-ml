"""
evaluate.py — Validation des performances du modèle.

Ce script lit les métriques sauvegardées et vérifie que le modèle
dépasse les seuils minimum acceptables.
Si le modèle est trop mauvais, le pipeline CI/CD s'arrête ici.
"""

import json
import sys

# ─────────────────────────────────────────────────────────────
# SEUILS MINIMUM ACCEPTABLES
# En production, un modèle qui ne dépasse pas ces seuils
# ne doit PAS être déployé.
# ─────────────────────────────────────────────────────────────
THRESHOLDS = {
    "accuracy": 0.70,   # Au moins 70% de bonnes prédictions
    "f1_score": 0.60,   # F1-score au moins 0.60 (important pour classes déséquilibrées)
    "roc_auc": 0.75,    # ROC-AUC au moins 0.75 (qualité de discrimination)
}


def load_metrics(filepath: str) -> dict:
    """Charge les métriques depuis le fichier JSON généré par train.py."""
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"ERREUR : Fichier de métriques introuvable → {filepath}")
        print("Avez-vous bien exécuté src/train.py avant ?")
        sys.exit(1)


def validate_metrics(metrics: dict) -> bool:
    """
    Compare chaque métrique avec son seuil minimum.
    Retourne True si tout est OK, False sinon.
    """
    print("=" * 55)
    print("  VALIDATION DES PERFORMANCES DU MODÈLE")
    print("=" * 55)
    print(f"{'Métrique':<15} {'Valeur':>10} {'Seuil':>10} {'Statut':>10}")
    print("-" * 55)

    all_passed = True

    for metric_name, threshold in THRESHOLDS.items():
        value = metrics.get(metric_name)

        if value is None:
            print(f"ERREUR : Métrique '{metric_name}' manquante dans le fichier JSON.")
            all_passed = False
            continue

        passed = value >= threshold
        status = "✅ OK" if passed else "❌ ÉCHEC"

        if not passed:
            all_passed = False

        print(f"{metric_name:<15} {value:>10.4f} {threshold:>10.4f} {status:>10}")

    print("-" * 55)
    return all_passed


def main():
    metrics = load_metrics("models/metrics.json")
    all_passed = validate_metrics(metrics)

    print()
    if all_passed:
        print("✅ VALIDATION RÉUSSIE : Toutes les métriques dépassent les seuils.")
        print("   Le modèle est prêt pour le déploiement.")
        sys.exit(0)  # Code 0 = succès → le pipeline CI/CD continue
    else:
        print("❌ VALIDATION ÉCHOUÉE : Certaines métriques sont insuffisantes.")
        print("   Le modèle NE SERA PAS déployé.")
        sys.exit(1)  # Code 1 = erreur → le pipeline CI/CD s'arrête


if __name__ == "__main__":
    main()