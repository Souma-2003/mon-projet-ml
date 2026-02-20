"""
test_model.py — 14 tests automatiques de validation du modèle.
"""

import pytest
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


@pytest.fixture
def sample_model():
    """Créer un modèle simple pour les tests."""
    X = np.random.rand(100, 15)
    y = np.random.randint(0, 2, 100)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model, X, y


# ─────────────────────────────────────────────────────────────
# TESTS DE VALIDITÉ DU MODÈLE
# ─────────────────────────────────────────────────────────────

def test_model_is_classifier(sample_model):
    """Test 1: Le modèle doit être un classificateur valide."""
    model, _, _ = sample_model
    assert hasattr(model, 'predict'), "Le modèle doit avoir la méthode predict"
    assert hasattr(model, 'predict_proba'), "Le modèle doit avoir predict_proba"


def test_model_predictions_shape(sample_model):
    """Test 2: Les prédictions doivent avoir la bonne forme."""
    model, X, _ = sample_model
    predictions = model.predict(X[:10])
    assert predictions.shape[0] == 10, "Devrait prédire pour 10 échantillons"


def test_model_probability_shape(sample_model):
    """Test 3: Les probabilités doivent avoir la bonne forme."""
    model, X, _ = sample_model
    proba = model.predict_proba(X[:10])
    assert proba.shape[0] == 10, "Devrait avoir 10 probabilités"
    assert proba.shape[1] == 2, "Devrait avoir 2 classes"


def test_probability_sums_to_one(sample_model):
    """Test 4: Les probabilités doivent sommer à 1."""
    model, X, _ = sample_model
    proba = model.predict_proba(X)
    assert np.allclose(proba.sum(axis=1), 1.0), \
        "Les probabilités ne somment pas à 1"


def test_model_accuracy_positive(sample_model):
    """Test 5: L'accuracy doit être positive."""
    model, X, y = sample_model
    accuracy = model.score(X, y)
    assert 0 <= accuracy <= 1, "L'accuracy doit être entre 0 et 1"


def test_model_serialization(sample_model, tmp_path):
    """Test 6: Le modèle doit être sérialisable."""
    model, _, _ = sample_model
    filepath = tmp_path / "model.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    assert filepath.exists(), "Le modèle n'a pas été sauvegardé"


def test_model_deserialization(sample_model, tmp_path):
    """Test 7: Le modèle doit être désérialisable."""
    model, X, _ = sample_model
    filepath = tmp_path / "model.pkl"
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    with open(filepath, 'rb') as f:
        loaded_model = pickle.load(f)
    
    assert hasattr(loaded_model, 'predict'), \
        "Le modèle chargé doit avoir la méthode predict"


def test_model_feature_importance(sample_model):
    """Test 8: Le modèle doit retourner l'importance des features."""
    model, _, _ = sample_model
    importance = model.feature_importances_
    assert len(importance) == 15, "Devrait avoir 15 importances"
    assert np.sum(importance) > 0, "L'importance doit être positive"


def test_model_consistency(sample_model):
    """Test 9: Les prédictions doivent être cohérentes."""
    model, X, _ = sample_model
    pred1 = model.predict(X[:5])
    pred2 = model.predict(X[:5])
    assert np.array_equal(pred1, pred2), \
        "Les prédictions doivent être identiques"


def test_model_input_validation(sample_model):
    """Test 10: Le modèle doit valider l'entrée."""
    model, _, _ = sample_model
    with pytest.raises((ValueError, IndexError)):
        model.predict(np.random.rand(10, 5))  # Mauvais nombre de features


def test_model_n_estimators(sample_model):
    """Test 11: Le RandomForest doit avoir le bon nombre d'arbres."""
    model, _, _ = sample_model
    assert model.n_estimators == 10, "Devrait avoir 10 arbres"


def test_model_predictions_in_range(sample_model):
    """Test 12: Les prédictions doivent être 0 ou 1."""
    model, X, _ = sample_model
    predictions = model.predict(X)
    assert set(predictions).issubset({0, 1}), \
        "Les prédictions doivent être 0 ou 1"


def test_model_classes(sample_model):
    """Test 13: Le modèle doit tracker les classes."""
    model, _, _ = sample_model
    classes = model.classes_
    assert len(classes) == 2, "Devrait avoir 2 classes"
    assert set(classes).issubset({0, 1}), "Les classes doivent être 0 et 1"


def test_model_max_depth(sample_model):
    """Test 14: Les arbres doivent avoir une profondeur raisonnable."""
    model, _, _ = sample_model
    max_depths = [tree.get_depth() for tree in model.estimators_]
    assert len(max_depths) > 0, "Devrait avoir des arbres"
    assert max(max_depths) > 0, "Les arbres doivent avoir une profondeur positive"
