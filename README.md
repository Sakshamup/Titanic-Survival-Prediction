# Titanic-Survival-Prediction

📌 Task Objectives
1) Perform data preprocessing and train multiple models to predict Titanic survival.
2) Use hyperparameter tuning and model evaluation to identify the best-performing model.
3) Save the best model for future use.

🚀 Steps to Run the Project
1) Clone the Repository
   ```
   git clone <your-repo-url>
   ```
3) Navigate to the Project Directory
   ```
   cd <repository-name>
   ```
5) Install Required Dependencies
    i) Ensure you have Python installed. Then, install the necessary libraries by running:
      ```
      pip install -r requirements.txt
      ```

6) Run the Model Training Script
      Execute the main Python script to preprocess the data, train models, and save the best-performing model:
    ```
   python app.py
   ```

5) Model Output
     i) The model training process will:
          Print accuracy, precision, recall, and F1-score metrics.
          Display the confusion matrix and classification report.
          Save the best-performing model as: titanic_best_model.pkl

⚙️ Technologies Used
1) Python
2) Libraries:
     pandas – for data manipulation
     numpy – for numerical operations
     scikit-learn – for model building and evaluation
     joblib – for saving the best model

✅ Model Evaluation
  After running the script, you will see the evaluation metrics for the best-performing model:
  Accuracy: Overall model correctness
  Precision: Correctly predicted positive cases out of all positive predictions
  Recall: Correctly predicted positive cases out of all actual positives
  F1 Score: Harmonic mean of precision and recall
  Confusion Matrix and Classification Report for detailed performance insights



