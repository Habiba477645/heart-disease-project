from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
import joblib
import pandas as pd

# Load data
df = pd.read_csv('C:/Users/ASUS/Downloads/habibas/IpProject2/data/heart.csv')

df.fillna({'restecg': df['restecg'].mode()[0]}, inplace=True)
df.fillna({'oldpeak': df['oldpeak'].mean()}, inplace=True)

df.drop_duplicates(inplace=True)


X = df.drop('target', axis=1)
y = df['target']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# Scale properly
scaler = MinMaxScaler()
X_train = X_train.copy()
X_test = X_test.copy()

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# Model (better config)
model = DecisionTreeClassifier(
    random_state=42,
    class_weight='balanced'
)

# Grid search
param_grid = {
    'max_depth': [3, 5, 8, 10],
    'min_samples_split': [2, 5, 10]
}

grid = GridSearchCV(model, param_grid, cv=5)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

# Evaluation
y_pred = best_model.predict(X_test)
acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)

print(f"Accuracy: {acc*100:.2f}%")
print(f"Precision: {prec*100:.2f}%")
print(f"Recall: {rec*100:.2f}%")
print(f"F1: {f1*100:.2f}%")
# Save
joblib.dump(best_model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Saved successfully")

