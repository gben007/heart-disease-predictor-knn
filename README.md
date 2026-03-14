# Heart Disease Predictor KNN

This is a heart disease prediction project built with KNN and Streamlit.

I first worked on the model in a notebook and then created a Streamlit app so the prediction can be used through a simple form instead of running everything in Jupyter.

## What this project does

- takes patient details and medical measurements as input
- uses a trained KNN model to predict heart disease risk
- shows the result in a simple Streamlit interface

## Files in this project

- `app.py` - Streamlit app
- `heart.csv` - dataset used in the project
- `knn_optimal_model.pkl` - saved KNN model
- `KNN_Classifier_Que_(1).ipynb` - notebook used for model building
- `requirements.txt` - required Python libraries
- `.gitignore` - ignores local files like virtual environment files

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
streamlit run app.py
```

After that, open the local URL shown in the terminal.

## About the app

The Streamlit app has a simple form where each field is clearly labeled, so it is easier to understand what value is being entered.

The app:

- collects patient information
- converts the inputs into the format expected by the model
- applies the same scaling logic used in the notebook
- makes a prediction using the saved KNN model

## Notes

- the model file used here is `knn_optimal_model.pkl`
- the scaler is rebuilt in the app using the dataset and the same training split logic
- this project is for learning and demonstration purposes

## Disclaimer

This project is not a medical tool and should not be used for actual diagnosis or treatment.
