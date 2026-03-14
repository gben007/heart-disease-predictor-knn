# Heart Disease Predictor KNN

A Streamlit web application for predicting heart disease risk using a trained K-Nearest Neighbors classifier.

This project wraps a notebook-trained machine learning model in a clean interactive interface so users can enter patient information, review medical measurements, and generate a prediction in a more understandable way.

## Overview

The application:

- loads a trained KNN model from `knn_optimal_model.pkl`
- rebuilds the feature scaler from `heart.csv`
- accepts 13 clinical and demographic inputs
- predicts whether a patient profile is closer to positive or negative heart disease cases
- presents the result through a simplified Streamlit UI

## Tech Stack

- Python
- Streamlit
- pandas
- scikit-learn

## Project Structure

```text
.
├── app.py
├── heart.csv
├── knn_optimal_model.pkl
├── KNN_Classifier_Que_(1).ipynb
├── requirements.txt
└── README.md
```

## Input Features

The model uses the following features:

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

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app will usually be available at:

```text
http://localhost:8501
```

## Notes On Preprocessing

The saved model is a trained KNN classifier. Because the original notebook used `StandardScaler`, the application rebuilds the scaler from the dataset using the same training split before running predictions.

## Recommended Improvements

If you want to evolve this into a stronger production-style portfolio project later, the next good steps would be:

- add unit tests for preprocessing and prediction flow
- save a full pipeline instead of saving only the classifier
- add model evaluation visuals
- add deployment configuration for Streamlit Community Cloud

## Disclaimer

This project is for educational purposes only and should not be used as a substitute for medical diagnosis or treatment.
