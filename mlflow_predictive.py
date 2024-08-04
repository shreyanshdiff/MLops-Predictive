import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score, f1_score, classification_report, recall_score
import joblib
import mlflow
import mlflow.sklearn

# Load data
df = pd.read_csv('predictive_maintenance.csv')

# Drop unnecessary columns
columns_to_drop = ['UDI', 'Product ID']
df.drop(columns=columns_to_drop, axis=1, inplace=True)

# Load existing encoders
encoder = joblib.load('type_encoder.pkl')
le = joblib.load('failure_type.pkl')

# Encode categorical variables
df['Type'] = encoder.transform(df['Type'])
df['Failure Type'] = le.transform(df['Failure Type'])

# Normalize numerical features
columns_to_normalize = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
scaler = MinMaxScaler()
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

# Split data into training and testing sets
x = df.drop('Failure Type', axis=1)
y = df['Failure Type']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Define parameter grid for RandomForest
param_grid = {'n_estimators': [10, 20, 50, 100, 200]}

model = RandomForestClassifier(random_state=42)
grid_param = GridSearchCV(param_grid=param_grid, verbose=5, estimator=model, cv=5, n_jobs=1)

with mlflow.start_run():
    grid_param.fit(x_train, y_train)

    mlflow.log_param("param_grid", param_grid)

    best_params = grid_param.best_params_
    mlflow.log_params(best_params)

    best_model = grid_param.best_estimator_

    pred = best_model.predict(x_test)
    accuracy = accuracy_score(y_test, pred)
    r2 = r2_score(y_test, pred)
    f1 = f1_score(y_test, pred, average='macro')
    recall = recall_score(y_test, pred, average='macro')

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("r2_score", r2)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("recall_score", recall)

    mlflow.sklearn.log_model(best_model, "model")
    mlflow.log_artifact("type_encoder.pkl")
    mlflow.log_artifact("failure_type.pkl")

    print(f"Best parameters: {best_params}")
    print(f"Test accuracy: {accuracy}")
    print(f"R2 score: {r2}")
    print(f"F1 score: {f1}")
    print(f"Recall score: {recall}")

joblib.dump(best_model, 'rfc_best_model.pkl')
