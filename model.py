import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

from preprocessing import prepare_data


def train_models():
    """Train and evaluate classification models."""

    data = prepare_data()

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=5,
            random_state=42
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    }

    results = {}

    for name, model in models.items():

        if name == "Logistic Regression":
            X_train = data["X_train_scaled"]
            X_test = data["X_test_scaled"]
        else:
            X_train = data["X_train"]
            X_test = data["X_test"]

        model.fit(X_train, data["y_train"])

        predictions = model.predict(X_test)

        results[name] = {
            "model": model,
            "accuracy": accuracy_score(
                data["y_test"], predictions
            ),
            "precision": precision_score(
                data["y_test"],
                predictions,
                average="weighted",
                zero_division=0
            ),
            "recall": recall_score(
                data["y_test"],
                predictions,
                average="weighted",
                zero_division=0
            ),
            "f1": f1_score(
                data["y_test"],
                predictions,
                average="weighted",
                zero_division=0
            ),
            "confusion_matrix": confusion_matrix(
                data["y_test"],
                predictions,
                labels=np.arange(len(data["class_names"]))
            ),
            "classification_report": classification_report(
                data["y_test"],
                predictions,
                target_names=data["class_names"],
                zero_division=0,
                output_dict=True
            ),
            "predictions": predictions
        }

    return data, results


def predict_species(
    model,
    measurements,
    encoder,
    scaler=None
):
    """Predict flower species and class probabilities."""

    features = np.asarray(
        measurements,
        dtype=float
    ).reshape(1, -1)

    if scaler is not None:
        features = scaler.transform(features)

    prediction = model.predict(features)[0]

    probabilities = model.predict_proba(features)[0]

    species = encoder.inverse_transform(
        [prediction]
    )[0]

    confidence = float(
        np.max(probabilities) * 100
    )

    probability_distribution = {
        species_name: float(probability * 100)
        for species_name, probability in zip(
            encoder.classes_,
            probabilities
        )
    }

    return {
        "species": species,
        "confidence": confidence,
        "probabilities": probability_distribution
    }