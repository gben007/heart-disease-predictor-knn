from pathlib import Path

import joblib
import pandas as pd


APP_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = APP_DIR / "heart.csv"
PIPELINE_PATH = APP_DIR / "heart_disease_pipeline.joblib"
LEGACY_MODEL_PATH = APP_DIR / "knn_optimal_model.pkl"
FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]
TARGET_COLUMN = "target"

SEX_OPTIONS = {"Female": 0, "Male": 1}
YES_NO_OPTIONS = {"No": 0, "Yes": 1}
CHEST_PAIN_OPTIONS = {
    "Typical angina": 0,
    "Atypical angina": 1,
    "Non-anginal pain": 2,
    "No chest pain symptoms": 3,
}
REST_ECG_OPTIONS = {
    "Normal": 0,
    "ST-T wave abnormality": 1,
    "Left ventricular hypertrophy": 2,
}
SLOPE_OPTIONS = {
    "Upsloping": 0,
    "Flat": 1,
    "Downsloping": 2,
}
THAL_OPTIONS = {
    "Normal": 0,
    "Fixed defect": 1,
    "Reversible defect": 2,
    "Other result": 3,
}


def load_dataset(data_path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(data_path)


def build_input_dataframe(
    *,
    age: int,
    sex_label: str,
    chest_pain_label: str,
    resting_bp: int,
    cholesterol: int,
    fasting_sugar_label: str,
    rest_ecg_label: str,
    max_heart_rate: int,
    exercise_angina_label: str,
    st_depression: float,
    slope_label: str,
    major_vessels: int,
    thal_label: str,
) -> pd.DataFrame:
    row = {
        "age": age,
        "sex": SEX_OPTIONS[sex_label],
        "cp": CHEST_PAIN_OPTIONS[chest_pain_label],
        "trestbps": resting_bp,
        "chol": cholesterol,
        "fbs": YES_NO_OPTIONS[fasting_sugar_label],
        "restecg": REST_ECG_OPTIONS[rest_ecg_label],
        "thalach": max_heart_rate,
        "exang": YES_NO_OPTIONS[exercise_angina_label],
        "oldpeak": st_depression,
        "slope": SLOPE_OPTIONS[slope_label],
        "ca": major_vessels,
        "thal": THAL_OPTIONS[thal_label],
    }
    return pd.DataFrame([row])[FEATURE_COLUMNS]


def load_pipeline(pipeline_path: Path = PIPELINE_PATH):
    return joblib.load(pipeline_path)
