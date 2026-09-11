from pathlib import Path
import json

import joblib
import pandas as pd
from clearml import Task

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = ROOT / "data" / "processed" / "train.csv"
TEST_PATH = ROOT / "data" / "processed" / "test.csv"
MODEL_PATH = ROOT / "models" / "penguins_model.joblib"
METRICS_PATH = ROOT / "metrics" / "metrics.json"

TARGET = "species"
RANDOM_STATE = 42


# Load processed data
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

X_train = train.drop(columns=[TARGET])
y_train = train[TARGET]

X_test = test.drop(columns=[TARGET])
y_test = test[TARGET]


# Feature engineering
numeric_features = X_train.select_dtypes(include="number").columns.tolist()
categorical_features = X_train.select_dtypes(exclude="number").columns.tolist()

preprocessor = ColumnTransformer([
    ("numeric", StandardScaler(), numeric_features),
    ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
])


# Create model
model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
    ),
])


# Train model
print("Training Logistic Regression...")
model.fit(X_train, y_train)


# Evaluate model
predictions = model.predict(X_test)

metrics = {
    "accuracy": float(accuracy_score(y_test, predictions)),
    "precision_macro": float(
        precision_score(y_test, predictions, average="macro", zero_division=0)
    ),
    "recall_macro": float(
        recall_score(y_test, predictions, average="macro", zero_division=0)
    ),
    "f1_macro": float(
        f1_score(y_test, predictions, average="macro", zero_division=0)
    ),
}

print("\nTest metrics:")
for name, value in metrics.items():
    print(f"{name}: {value:.4f}")


# Save model
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(
    {
        "model": model,
        "label_encoder": None,
    },
    MODEL_PATH,
)


# Save metrics
METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(METRICS_PATH, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)


# Log experiment in ClearML
task = Task.init(
    project_name="PMLDL Assignment 1",
    task_name="Logistic Regression Training",
)

task.connect({
    "C": 1.0,
    "class_weight": "balanced",
    "max_iter": 2000,
    "random_state": RANDOM_STATE,
})

logger = task.get_logger()

for name, value in metrics.items():
    logger.report_single_value(name, value)

task.upload_artifact("trained_model", artifact_object=MODEL_PATH)
task.upload_artifact("test_metrics", artifact_object=metrics)

task.close()

print("\nModel saved:", MODEL_PATH)
print("Metrics saved:", METRICS_PATH)
print("Stage 2 completed successfully.")