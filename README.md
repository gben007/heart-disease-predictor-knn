# Heart Disease Predictor KNN

This project predicts heart disease risk using a K-Nearest Neighbors model and a Streamlit web app.

I first built the model in a notebook, then moved the project into a cleaner app structure so it is easier to run, test, and deploy.

## What this project does

- takes patient details and medical measurements as input
- uses a trained KNN model to predict heart disease risk
- shows the result in a Streamlit interface

## Project files

- `app.py` - Streamlit app for prediction
- `train.py` - training script that creates the serving-ready model artifact
- `src/heart_predictor.py` - shared project logic for input mapping and loading the model
- `tests/test_heart_predictor.py` - small test file for core functionality
- `heart.csv` - dataset used in the project
- `heart_disease_pipeline.joblib` - saved pipeline used by the app
- `knn_optimal_model.pkl` - original saved model from the notebook
- `KNN_Classifier_Que_(1).ipynb` - notebook used during model work
- `requirements.txt` - required Python libraries
- `.gitignore` - ignores local files

## Input features used

The model uses these features:

- `age`
- `sex`
- `cp`
- `trestbps`
- `chol`
- `fbs`
- `restecg`
- `thalach`
- `exang`
- `oldpeak`
- `slope`
- `ca`
- `thal`

## How to run the project

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
streamlit run app.py
```

After that, open the local URL shown in the terminal.

## Why `train.py` was added

Earlier the app rebuilt scaling directly from the dataset when it started. That works, but it is not a strong way to serve a model.

Now `train.py` creates one pipeline artifact that contains:

- scaling
- the KNN model
- training metadata

This makes the app simpler and closer to a proper deployment setup because the app only loads one saved artifact.

## Tests

The project now includes a small test file to check:

- whether form inputs are converted correctly
- whether the saved pipeline artifact can load and make a prediction

You can run the tests with:

```bash
pytest
```

## Notes

- the app uses `heart_disease_pipeline.joblib` as the main serving artifact
- `knn_optimal_model.pkl` is still kept because it came from the original notebook work
- this project is for learning and demonstration purposes

## Disclaimer

This project is not a medical tool and should not be used for actual diagnosis or treatment.
