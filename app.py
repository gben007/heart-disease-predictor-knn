import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="Heart",
    layout="wide",
)


APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "knn_optimal_model.pkl"
DATA_PATH = APP_DIR / "heart.csv"
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


st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top right, rgba(42, 111, 151, 0.10), transparent 28%),
            linear-gradient(180deg, #f8fbfd 0%, #edf3f8 100%);
        color: #162033;
    }

    .block-container {
        max-width: 1100px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    [data-testid="stSidebar"] {
        display: none;
    }

    .hero {
        background: linear-gradient(135deg, #12304b 0%, #1f537d 100%);
        border-radius: 26px;
        padding: 2rem 2rem 1.8rem 2rem;
        color: white;
        box-shadow: 0 20px 42px rgba(18, 48, 75, 0.18);
        margin-bottom: 1.25rem;
    }

    .hero-kicker {
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.8rem;
        opacity: 0.8;
        margin-bottom: 0.7rem;
    }

    .hero h1 {
        margin: 0;
        font-size: 2.45rem;
        line-height: 1.04;
        color: white;
    }

    .hero p {
        margin: 0.8rem 0 0 0;
        max-width: 58ch;
        font-size: 1rem;
        opacity: 0.94;
    }

    .surface {
        background: rgba(255, 255, 255, 0.96);
        border: 1px solid rgba(22, 32, 51, 0.08);
        border-radius: 24px;
        padding: 1.4rem;
        box-shadow: 0 14px 30px rgba(22, 32, 51, 0.06);
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 800;
        color: #162033;
        margin-bottom: 0.25rem;
    }

    .section-copy {
        color: #5d6b7d;
        font-size: 0.96rem;
        margin-bottom: 1rem;
    }

    .group-title {
        font-size: 1rem;
        font-weight: 800;
        color: #173b61;
        margin-top: 0.4rem;
        margin-bottom: 0.8rem;
        padding-bottom: 0.45rem;
        border-bottom: 1px solid rgba(23, 59, 97, 0.10);
    }

    .field-card {
        background: #f8fbfe;
        border: 1px solid rgba(29, 79, 122, 0.09);
        border-radius: 18px;
        padding: 0.85rem 0.9rem 0.65rem 0.9rem;
        margin-bottom: 0.9rem;
    }

    .field-label {
        font-size: 0.96rem;
        font-weight: 800;
        color: #162033;
        margin-bottom: 0.2rem;
    }

    .field-help {
        font-size: 0.85rem;
        color: #667588;
        margin-bottom: 0.6rem;
        line-height: 1.4;
    }

    .tip-box {
        background: #f7fbff;
        border: 1px solid rgba(23, 59, 97, 0.10);
        border-radius: 20px;
        padding: 1rem;
    }

    .tip-box h3 {
        margin: 0 0 0.55rem 0;
        color: #173b61;
        font-size: 1rem;
    }

    .tip-box p {
        margin: 0.35rem 0;
        color: #5d6b7d;
        font-size: 0.93rem;
    }

    .result-safe,
    .result-risk {
        border-radius: 22px;
        padding: 1.25rem;
        margin-top: 1rem;
    }

    .result-safe {
        background: linear-gradient(135deg, #edf9f2 0%, #f9fffc 100%);
        border: 1px solid rgba(29, 123, 83, 0.18);
    }

    .result-risk {
        background: linear-gradient(135deg, #fff2ef 0%, #fffaf8 100%);
        border: 1px solid rgba(187, 63, 45, 0.18);
    }

    .result-chip {
        display: inline-block;
        padding: 0.34rem 0.72rem;
        border-radius: 999px;
        font-size: 0.83rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
    }

    .result-safe .result-chip {
        background: #d7f1e3;
        color: #1b714f;
    }

    .result-risk .result-chip {
        background: #ffd9d2;
        color: #b13f2d;
    }

    .result-title {
        margin: 0;
        color: #162033;
        font-size: 1.45rem;
        font-weight: 800;
    }

    .result-text {
        color: #5d6b7d;
        margin-top: 0.6rem;
        margin-bottom: 0;
    }

    .confidence {
        display: inline-block;
        margin-top: 0.85rem;
        padding: 0.55rem 0.8rem;
        background: rgba(22, 32, 51, 0.06);
        color: #162033;
        font-weight: 700;
        border-radius: 12px;
    }

    .stButton > button,
    .stForm button {
        width: 100%;
        min-height: 3.2rem;
        border-radius: 14px;
        border: none !important;
        background: linear-gradient(135deg, #1d4f7a, #173b61) !important;
        color: #ffffff !important;
        font-size: 1rem;
        font-weight: 800;
        box-shadow: 0 12px 22px rgba(29, 79, 122, 0.18);
    }

    .stButton > button:hover,
    .stButton > button:focus,
    .stButton > button:active,
    .stForm button:hover,
    .stForm button:focus,
    .stForm button:active {
        background: linear-gradient(135deg, #1d4f7a, #173b61) !important;
        color: #ffffff !important;
        border: none !important;
        box-shadow: 0 12px 22px rgba(29, 79, 122, 0.18) !important;
    }

    @media (max-width: 900px) {
        .hero h1 {
            font-size: 2rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_dataset(data_path: Path) -> pd.DataFrame:
    return pd.read_csv(data_path)


@st.cache_resource
def load_model(model_path: Path):
    with open(model_path, "rb") as file:
        return pickle.load(file)


@st.cache_resource
def load_artifacts(model_path: Path, data_path: Path):
    dataset = load_dataset(data_path)
    X = dataset[FEATURE_COLUMNS]
    y = dataset["target"]
    X_train, _, _, _ = train_test_split(X, y, test_size=0.33, random_state=42)

    scaler = StandardScaler()
    scaler.fit(X_train)

    model = load_model(model_path)
    return scaler, model


def build_input_dataframe(
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
    return pd.DataFrame(
        [
            {
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
        ]
    )[FEATURE_COLUMNS]


def field_label(title: str, help_text: str) -> None:
    st.markdown(
        f"""
        <div class="field-card">
            <div class="field-label">{title}</div>
            <div class="field-help">{help_text}</div>
        """,
        unsafe_allow_html=True,
    )


def field_label_end() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


if not MODEL_PATH.exists():
    st.error(f"Missing model file: `{MODEL_PATH}`")
    st.stop()

if not DATA_PATH.exists():
    st.error(f"Missing dataset file: `{DATA_PATH}`")
    st.stop()

try:
    scaler, model = load_artifacts(MODEL_PATH, DATA_PATH)
except Exception as exc:
    st.error(f"Could not load the app assets: {exc}")
    st.stop()


st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">Clear Form Design</div>
        <h1>Heart Disease Predictor</h1>
        <p>
            Every input below has its own heading directly above it, with a short explanation.
            Fill the form and then click the button to get the result.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

main_col, side_col = st.columns([1.55, 0.85], gap="large")

with main_col:
    st.markdown('<div class="surface">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Patient Information Form</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Each detail is shown with a clear label above the input box.</div>',
        unsafe_allow_html=True,
    )

    with st.form("heart_form"):
        st.markdown('<div class="group-title">Basic Details</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2, gap="large")

        with col1:
            field_label("Age", "Enter the age of the patient in years.")
            age = st.number_input("Age", min_value=29, max_value=77, value=55, label_visibility="collapsed")
            field_label_end()

            field_label("Sex", "Choose the sex of the patient.")
            sex_label = st.selectbox("Sex", list(SEX_OPTIONS.keys()), label_visibility="collapsed")
            field_label_end()

            field_label("Chest Pain Type", "Select the type of chest pain the patient has.")
            chest_pain_label = st.selectbox(
                "Chest Pain Type",
                list(CHEST_PAIN_OPTIONS.keys()),
                label_visibility="collapsed",
            )
            field_label_end()

        with col2:
            field_label("Pain During Exercise", "Choose Yes if chest pain happens during exercise.")
            exercise_angina_label = st.selectbox(
                "Pain During Exercise",
                list(YES_NO_OPTIONS.keys()),
                label_visibility="collapsed",
            )
            field_label_end()

            field_label("Maximum Heart Rate Achieved", "Enter the highest heart rate reached during testing.")
            max_heart_rate = st.number_input(
                "Maximum Heart Rate Achieved",
                min_value=71,
                max_value=202,
                value=153,
                label_visibility="collapsed",
            )
            field_label_end()

            field_label("Thalassemia Test Result", "Select the test result category.")
            thal_label = st.selectbox(
                "Thalassemia Test Result",
                list(THAL_OPTIONS.keys()),
                label_visibility="collapsed",
            )
            field_label_end()

        st.markdown('<div class="group-title">Measurements</div>', unsafe_allow_html=True)
        col3, col4 = st.columns(2, gap="large")

        with col3:
            field_label("Resting Blood Pressure", "Enter resting blood pressure in mm Hg.")
            resting_bp = st.number_input(
                "Resting Blood Pressure",
                min_value=94,
                max_value=200,
                value=130,
                label_visibility="collapsed",
            )
            field_label_end()

            field_label("Serum Cholesterol", "Enter cholesterol value in mg/dl.")
            cholesterol = st.number_input(
                "Serum Cholesterol",
                min_value=126,
                max_value=564,
                value=240,
                label_visibility="collapsed",
            )
            field_label_end()

            field_label("Fasting Blood Sugar Above 120 mg/dl", "Choose Yes if the fasting blood sugar is above 120 mg/dl.")
            fasting_sugar_label = st.selectbox(
                "Fasting Blood Sugar Above 120 mg/dl",
                list(YES_NO_OPTIONS.keys()),
                label_visibility="collapsed",
            )
            field_label_end()

        with col4:
            field_label("Resting ECG Result", "Choose the resting ECG finding.")
            rest_ecg_label = st.selectbox(
                "Resting ECG Result",
                list(REST_ECG_OPTIONS.keys()),
                label_visibility="collapsed",
            )
            field_label_end()

            field_label("Slope of Exercise ST Segment", "Choose the slope shown in the stress test.")
            slope_label = st.selectbox(
                "Slope of Exercise ST Segment",
                list(SLOPE_OPTIONS.keys()),
                label_visibility="collapsed",
            )
            field_label_end()

            field_label("Number of Major Vessels Seen", "Select how many major vessels were seen by fluoroscopy.")
            major_vessels = st.selectbox(
                "Number of Major Vessels Seen",
                [0, 1, 2, 3, 4],
                label_visibility="collapsed",
            )
            field_label_end()

        st.markdown('<div class="group-title">Stress Test Reading</div>', unsafe_allow_html=True)
        field_label("ST Depression Value", "Enter the ST depression value, also called oldpeak in the dataset.")
        st_depression = st.number_input(
            "ST Depression Value",
            min_value=0.0,
            max_value=6.2,
            value=1.0,
            step=0.1,
            label_visibility="collapsed",
        )
        field_label_end()

        submitted = st.form_submit_button("Predict Heart Disease Risk")

    st.markdown("</div>", unsafe_allow_html=True)

with side_col:
    st.markdown(
        """
        <div class="tip-box">
            <h3>What you are filling</h3>
            <p><strong>Basic Details:</strong> age, sex, chest pain, pain during exercise, heart rate, and thalassemia result.</p>
            <p><strong>Measurements:</strong> blood pressure, cholesterol, blood sugar, ECG result, ST slope, and major vessels.</p>
            <p><strong>Stress Test Reading:</strong> the ST depression value from the exercise test.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


input_df = build_input_dataframe(
    age=age,
    sex_label=sex_label,
    chest_pain_label=chest_pain_label,
    resting_bp=resting_bp,
    cholesterol=cholesterol,
    fasting_sugar_label=fasting_sugar_label,
    rest_ecg_label=rest_ecg_label,
    max_heart_rate=max_heart_rate,
    exercise_angina_label=exercise_angina_label,
    st_depression=st_depression,
    slope_label=slope_label,
    major_vessels=major_vessels,
    thal_label=thal_label,
)

if submitted:
    scaled_input = scaler.transform(input_df)
    prediction = int(model.predict(scaled_input)[0])
    probabilities = model.predict_proba(scaled_input)[0]
    confidence = float(probabilities[prediction])

    if prediction == 1:
        st.markdown(
            f"""
            <div class="result-risk">
                <div class="result-chip">Higher Risk Pattern</div>
                <p class="result-title">This profile is closer to heart disease cases.</p>
                <p class="result-text">
                    The answers entered in the form are more similar to positive heart disease examples in the dataset.
                </p>
                <div class="confidence">Confidence: {confidence:.2%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="result-safe">
                <div class="result-chip">Lower Risk Pattern</div>
                <p class="result-title">This profile is closer to non-disease cases.</p>
                <p class="result-text">
                    The answers entered in the form are more similar to negative heart disease examples in the dataset.
                </p>
                <div class="confidence">Confidence: {confidence:.2%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    st.info("Fill the form and click the button to get the prediction.")


st.caption("Educational use only. This app is not a medical diagnosis tool.")
