# Heart Disease Detection System

A hybrid heart disease risk prediction system using a Rule-Based Expert System (Experta) and a Machine Learning Decision Tree (Scikit-Learn).

---

## Project Structure

```
Heart_Disease_Detection/
├── data/
│   ├── heart.csv
│   └── cleaned_data.csv
├── notebooks/
│   ├── data_preprocessing.ipynb
│   └── data_visualization.ipynb
├── rule_based_system/
│   ├── rules.py
│   └── ExprtSystem.py
├── ml_model/
│   ├── train_model.py
│   └── predict.py
├── reports/
│   └── accuracy_comparison.md
├── ui/
│   └── app.py
├── README.md
└── requirements.txt
```

## Setup & Installation

```bash
pip install -r requirements.txt
```

## How to Run

**1. Preprocess the data:**
```bash
jupyter notebook notebooks/data_preprocessing.ipynb
```

**2. Train the model:**
```bash
python ml_model/train_model.py
```

**3. Run the Expert System evaluation:**
```bash
python rule_based_system/ExprtSystem.py
```

**4. Launch the Streamlit app:**
```bash
streamlit run ui/app.py
```

---

## Models

- **Expert System**: Rule-based inference engine using Experta with 10 medical rules
- **Decision Tree**: Trained with GridSearchCV hyperparameter tuning (80/20 split)

---

## Results

| Metric    | Decision Tree | Expert System |
|-----------|:------------:|:-------------:|
| Accuracy  | 85.25%       | 49.76%        |
| Precision | 80.00%      | 0.00%         |
| Recall    | 93.33%       | 0.00%         |
| F1 Score  | 86.15%       | 0.00%         |

---

## Dataset

- Features: age, sex, chest pain type, blood pressure, cholesterol, and more
- Target: 1 = heart disease, 0 = no heart disease