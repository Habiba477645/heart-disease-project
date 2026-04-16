import joblib
import pandas as pd

# Load model + scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')


features = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak',
    'slope', 'ca', 'thal'
]

num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']


def get_user_input(input: dict) -> pd.DataFrame:
    data = {}

    for feature in features:
        data[feature] = input.get(feature, 0)

    df = pd.DataFrame([data])

    # Apply SAME scaling
    df[num_cols] = scaler.transform(df[num_cols])

    return df


def predict(input: dict):
    df = get_user_input(input)
    return model.predict(df)[0]
