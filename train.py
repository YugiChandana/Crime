import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load the dataset
data = pd.read_csv('CRIME_DETAILS_RANDOMIZED.csv')

# Fix column names6
data.columns = data.columns.str.strip()

# Print available columns
print("Available Columns:", data.columns.tolist())

# Detect crime columns (excluding non-crime)
non_crime_columns = ['Id', 'State', 'Year', 'Arrested']
crime_columns = [col for col in data.columns if col not in non_crime_columns]

print("Crime Columns Detected:", crime_columns)

# Save crime columns list
joblib.dump(crime_columns, 'crime_columns.pkl')

# Label Encode State and Year
label_encoders = {}
for column in ['State', 'Year']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column].astype(str))
    label_encoders[column] = le

# Save the label encoders
joblib.dump(label_encoders, 'label_encoders.pkl')

# Example: Choose 1 crime column for default model (e.g., 'Rape')
default_crime = crime_columns[0]

# Feature and target
X = data[['State', 'Year', default_crime]]
y = data['Arrested']

# Train the model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)
model.fit(X, y)

# Save the model
joblib.dump(model, 'model.h5')

# Evaluation
y_pred = model.predict(X)

acc = accuracy_score(y, y_pred)
prec = precision_score(y, y_pred, pos_label='Yes')
rec = recall_score(y, y_pred, pos_label='Yes')

print("\nModel Evaluation:")
print(f"Accuracy: {acc:.2f}")
print(f"Precision: {prec:.2f}")
print(f"Recall: {rec:.2f}")
# Save model metrics
metrics = {
    'accuracy': acc,
    'precision': prec,
    'recall': rec
}
joblib.dump(metrics, 'metrics.pkl')

