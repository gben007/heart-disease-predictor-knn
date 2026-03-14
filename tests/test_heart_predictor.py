from src.heart_predictor import (
    FEATURE_COLUMNS,
    PIPELINE_PATH,
    build_input_dataframe,
    load_pipeline,
)


def test_build_input_dataframe_has_expected_columns():
    input_df = build_input_dataframe(
        age=55,
        sex_label="Male",
        chest_pain_label="Atypical angina",
        resting_bp=130,
        cholesterol=240,
        fasting_sugar_label="No",
        rest_ecg_label="Normal",
        max_heart_rate=153,
        exercise_angina_label="No",
        st_depression=1.0,
        slope_label="Flat",
        major_vessels=0,
        thal_label="Reversible defect",
    )

    assert list(input_df.columns) == FEATURE_COLUMNS
    assert input_df.loc[0, "sex"] == 1
    assert input_df.loc[0, "cp"] == 1
    assert input_df.loc[0, "thal"] == 2


def test_pipeline_artifact_can_predict():
    artifact = load_pipeline(PIPELINE_PATH)
    pipeline = artifact["pipeline"]

    input_df = build_input_dataframe(
        age=55,
        sex_label="Male",
        chest_pain_label="Atypical angina",
        resting_bp=130,
        cholesterol=240,
        fasting_sugar_label="No",
        rest_ecg_label="Normal",
        max_heart_rate=153,
        exercise_angina_label="No",
        st_depression=1.0,
        slope_label="Flat",
        major_vessels=0,
        thal_label="Reversible defect",
    )

    prediction = pipeline.predict(input_df)
    probabilities = pipeline.predict_proba(input_df)

    assert prediction.shape == (1,)
    assert probabilities.shape == (1, 2)
    assert 0.0 <= float(probabilities[0][0]) <= 1.0
    assert 0.0 <= float(probabilities[0][1]) <= 1.0
