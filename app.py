import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib

# Load dataset
# Ensure you have the titanic.csv file in the same directory or provide the correct path
data = pd.read_csv('/content/tested.csv')

# Data Preprocessing
# Handle missing values by filling NaN with most frequent values
imputer = SimpleImputer(strategy='most_frequent')
data[['Age', 'Fare']] = imputer.fit_transform(data[['Age', 'Fare']])
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# Encoding categorical variables using OneHotEncoder
categorical_features = ['Sex', 'Embarked']
numerical_features = ['Age', 'Fare', 'Pclass']

# Preprocessing pipeline for consistent data transformation
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),  # Scale numerical features
        ('cat', OneHotEncoder(), categorical_features)  # One-hot encode categorical features
    ]
)

# Splitting data into training and testing sets
X = data[['Age', 'Fare', 'Pclass', 'Sex', 'Embarked']]
y = data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model selection and hyperparameter tuning
models = {
    'RandomForest': RandomForestClassifier(),
    'GradientBoosting': GradientBoostingClassifier(),
    'AdaBoost': AdaBoostClassifier()
}

# Define hyperparameters for tuning
params = {
    'RandomForest': {'n_estimators': [100, 200], 'max_depth': [5, 10]},
    'GradientBoosting': {'n_estimators': [100, 150], 'learning_rate': [0.05, 0.1]},
    'AdaBoost': {'n_estimators': [50, 100], 'learning_rate': [0.5, 1.0]}
}

best_model = None
best_score = 0

# Iterating through each model for hyperparameter tuning
# Iterating through each model for hyperparameter tuning
for name, model in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),  # Apply preprocessing steps
        ('classifier', model)  
    ])

    # Grid search for best hyperparameters
    # Adjust parameter names for GridSearchCV to target the 'classifier' step in the pipeline
    grid_params = {f'classifier__{key}': value for key, value in params[name].items()}  
    grid = GridSearchCV(pipeline, grid_params, cv=5, scoring='accuracy')  
    grid.fit(X_train, y_train)


    # Model evaluation on test set
    y_pred = grid.best_estimator_.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Save the best performing model
    if acc > best_score:
        best_score = acc
        best_model = grid.best_estimator_

# Final model evaluation
# Making predictions with the best model
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Printing evaluation metrics
print("Best Model:", best_model)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the best model for future use
joblib.dump(best_model, 'titanic_best_model.pkl')

# Inline comments added for clarity and documentation
