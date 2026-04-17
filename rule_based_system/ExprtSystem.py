
import collections
import collections.abc

collections.Mapping = collections.abc.Mapping
from experta import Fact
from rules import HeartDiseaseRisk
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



def run_expert_system(age, trestbps, chol, thalach, oldpeak, exang_1):
    # Initialize and reset the inference engine
    engine = HeartDiseaseRisk()
    engine.reset()

    # Declare patient facts into the knowledge base
    engine.declare(Fact(
        age      = age,
        trestbps = trestbps,
        chol     = chol,
        thalach  = thalach,
        oldpeak  = oldpeak,
        exang_1  = exang_1
    ))

    # Fire all matching rules
    engine.run()

    counts = engine.counts
    fired  = engine.fired_rules

    # Determine final risk based on rule counts (highest priority wins)
    if counts["High"] > 0:
        final_risk = "High"
    elif counts["Medium"] > 0:
        final_risk = "Medium"
    else:
        final_risk = "Low"

    return {
        "final_risk" : final_risk,
        "fired_rules": fired,
        "counts"     : counts
    }


def evaluate_expert_system(data_path):
    

    # Load cleaned dataset
    df = pd.read_csv(data_path)
    X  = df.drop('target', axis=1)
    y  = df['target']

    # Use the same split as the Decision Tree for a fair comparison
    _, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Min-max normalization helper for raw feature values
    def normalize(val, mn, mx):
        return (val - mn) / (mx - mn) if mx != mn else 0.0

    # Run the expert system on each validation record
    es_preds = []
    for _, row in X_val.iterrows():
        result = run_expert_system(
            age      = normalize(row.get('age',      0), 20,  80),
            trestbps = normalize(row.get('trestbps', 0), 80, 200),
            chol     = normalize(row.get('chol',     0), 100, 400),
            thalach  = normalize(row.get('thalach',  0), 70, 210),
            oldpeak  = normalize(row.get('oldpeak',  0), 0,    6),
            exang_1  = bool(row.get('exang_1', 0)),
        )
        # Map High → 1 (disease), Medium/Low → 0 (no disease)
        es_preds.append(1 if result['final_risk'] == 'High' else 0)

    # Compute evaluation metrics
    acc  = accuracy_score (y_val, es_preds)
    prec = precision_score(y_val, es_preds, zero_division=0)
    rec  = recall_score   (y_val, es_preds, zero_division=0)
    f1   = f1_score       (y_val, es_preds, zero_division=0)

    print("EXPERT SYSTEM — Evaluation Results")
    print(f"Accuracy  : {acc  * 100:.2f}%")
    print(f"Precision : {prec * 100:.2f}%")
    print(f"Recall    : {rec  * 100:.2f}%")
    print(f"F1 Score  : {f1   * 100:.2f}%")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


#  Entry point
if __name__ == "__main__":
    BASE      = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(BASE, '..', 'data', 'cleaned_data.csv')
    evaluate_expert_system(data_path) 
