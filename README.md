# Predictive Maintenance: Failure Prediction

This project focuses on predicting failure types in machinery using a machine learning approach. The dataset contains various features related to the operating conditions and states of machines, and the goal is to predict the type of failure based on these features.

## Dataset

The dataset used in this project is `predictive_maintenance.csv`, which contains the following columns:

- `UDI`: Unique identifier for each observation (dropped during preprocessing).
- `Product ID`: Identifier for each product (dropped during preprocessing).
- `Type`: Categorical variable representing the type of product.
- `Air temperature [K]`: Continuous variable representing the air temperature in Kelvin.
- `Process temperature [K]`: Continuous variable representing the process temperature in Kelvin.
- `Rotational speed [rpm]`: Continuous variable representing the rotational speed in RPM.
- `Torque [Nm]`: Continuous variable representing the torque in Nm.
- `Tool wear [min]`: Continuous variable representing the tool wear in minutes.
- `Failure Type`: Target variable representing the type of failure.

## Preprocessing

1. **Dropping Unnecessary Columns**: The columns `UDI` and `Product ID` were removed as they are not needed for prediction.

2. **Encoding Categorical Variables**: 
   - The `Type` column was encoded using `LabelEncoder`.
   - The `Failure Type` column, which is the target, was also encoded using `LabelEncoder`.

3. **Normalizing Continuous Variables**: 
   - The features `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, and `Tool wear [min]` were normalized using `MinMaxScaler`.

## Model Training

- **Algorithm**: Random Forest Classifier
- **Hyperparameter Tuning**: Used `GridSearchCV` for tuning the number of estimators (`n_estimators`) with values `[10, 20, 50, 100, 200]`.
- **Data Splitting**: The data was split into training and testing sets using an 80-20 split.

## Model Evaluation

- **Metrics Used**:
  - Accuracy
  - R² Score
  - F1 Score (macro average)
  - Recall Score (macro average)
  - Classification Report

## Results

- **Best Hyperparameters**: The best number of estimators found using grid search.
- **Best Cross-Validation Score**: The highest accuracy score obtained during cross-validation.
- **Test Score**: Accuracy on the test data.
- **Classification Metrics**: Detailed evaluation using accuracy, R² score, F1 score, and recall score.

## Usage

To run this project, follow these steps:

1. Ensure you have the necessary Python libraries installed: `pandas`, `numpy`, `scikit-learn`, `joblib`.
2. Load the dataset and execute the preprocessing steps outlined above.
3. Train the Random Forest model using the specified hyperparameters.
4. Evaluate the model using the provided metrics.
5. Save the best model using `joblib` for future predictions.

## Saving and Loading the Model

- The best model is saved using `joblib` for future inference:
  ```python
  import joblib
  joblib.dump(best_model, 'rfc_best_model.pkl')
