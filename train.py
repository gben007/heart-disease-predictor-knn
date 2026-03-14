from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.heart_predictor import (
    DATA_PATH,
    FEATURE_COLUMNS,
    PIPELINE_PATH,
    TARGET_COLUMN,
    load_dataset,
)


RANDOM_STATE = 42
TEST_SIZE = 0.33
N_NEIGHBORS = 7


def train_and_save(
    data_path: Path = DATA_PATH,
    pipeline_path: Path = PIPELINE_PATH,
):
    dataset = load_dataset(data_path)
    X = dataset[FEATURE_COLUMNS]
    y = dataset[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=N_NEIGHBORS)),
        ]
    )
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    artifact = {
        "pipeline": pipeline,
        "feature_columns": FEATURE_COLUMNS,
        "metrics": {
            "accuracy": float(accuracy),
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "n_neighbors": N_NEIGHBORS,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
        },
    }
    joblib.dump(artifact, pipeline_path)
    return artifact


if __name__ == "__main__":
    artifact = train_and_save()
    metrics = artifact["metrics"]
    print("Saved pipeline artifact to:", PIPELINE_PATH)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print("Train rows:", metrics["train_rows"])
    print("Test rows:", metrics["test_rows"])
