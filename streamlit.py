import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved model
model = joblib.load('titanic_best_model.pkl')

# Streamlit app title and description
st.title("🚢 Titanic Survival Prediction")
st.write("""
Enter the passenger details to predict whether they survived or not.
""")

# Input fields for user data
age = st.number_input("Age", min_value=0, max_value=100, value=30)
fare = st.number_input("Fare", min_value=0.0, value=32.0)
pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
sex = st.selectbox("Sex", ['male', 'female'])
embarked = st.selectbox("Embarked", ['C', 'Q', 'S'])

# Button to trigger prediction
if st.button("Predict"):
    # Create a DataFrame with the raw inputs (5 features only)
    input_data = pd.DataFrame({
        'Age': [age],
        'Fare': [fare],
        'Pclass': [pclass],
        'Sex': [sex],
        'Embarked': [embarked]
    })

    # Perform prediction using the pre-trained model pipeline
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # Display the result
    if prediction == 1:
        st.success(f"✅ The passenger **survived** with a probability of {probability:.2%}")
    else:
        st.error(f"❌ The passenger **did not survive** with a probability of {1 - probability:.2%}")

# Add footer info
st.write("---")
st.write("🔍 *Titanic Survival Prediction model using Streamlit*")
