!pip install streamlit pandas joblib huggingface-hub -q

import streamlit as st
import pandas as pd
import joblib
import os
from huggingface_hub import hf_hub_download # Import hf_hub_download

# Use the HF_TOKEN from Colab secrets for authentication
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    st.error("Hugging Face token not found. Please add it to Colab secrets as 'HF_TOKEN'.")
else:
    try:
        model_path = hf_hub_download(repo_id="Priyanka-Ankam/WellnessPkgProject", filename="WellnessPkgProject.joblib", token=HF_TOKEN) # Pass the token
        model = joblib.load(model_path)

        # Streamlit UI for Product Purchase Prediction
        st.title("Product Purchase Prediction App")
        st.write("""
        This application predicts the likelihood of a customer purchasing the Wellness Tourism Package.
        Please enter the required data below to get a prediction.
        """)

        # User input fields based on the tourism.csv dataset columns
        st.header("Customer Information")
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        typeofcontact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
        citytier = st.selectbox("City Tier", [1, 2, 3])
        durationofpitch = st.number_input("Duration of Pitch (minutes)", min_value=0.0, value=10.0)
        occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer", "Government Sector"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        numberofpersonvisiting = st.number_input("NumberOfPersonVisiting", min_value=1, value=2)
        preferredpropertystar = st.selectbox("PreferredPropertyStar", [1.0, 2.0, 3.0, 4.0, 5.0])
        maritalstatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        numberoftrips = st.number_input("Number of Trips (annual)", min_value=0.0, value=1.0)
        passport = st.selectbox("Passport", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
        pitchsatisfactionscore = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
        owncar = st.selectbox("Own Car", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
        numberofchildrenvisiting = st.number_input("NumberOfChildrenVisiting", min_value=0.0, value=0.0)
        designation = st.selectbox("Designation", ["Manager", "Executive", "Senior Manager", "AVP", "VP", "Director"])
        monthlyincome = st.number_input("Monthly Income", min_value=0.0, value=20000.0)
        productpitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Super Deluxe", "King", "Standard"])


        # Assemble input into DataFrame
        input_data = pd.DataFrame([{
            'Age': age,
            'TypeofContact': typeofcontact,
            'CityTier': citytier,
            'DurationOfPitch': durationofpitch,
            'Occupation': occupation,
            'Gender': gender,
            'NumberOfPersonVisiting': numberofpersonvisiting,
            'PreferredPropertyStar': preferredpropertystar,
            'MaritalStatus': maritalstatus,
            'NumberOfTrips': numberoftrips,
            'Passport': passport,
            'PitchSatisfactionScore': pitchsatisfactionscore,
            'OwnCar': owncar,
            'NumberOfChildrenVisiting': numberofchildrenvisiting,
            'Designation': designation,
            'MonthlyIncome': monthlyincome,
            'ProductPitched': productpitched
        }])


        if st.button("Predict Purchase"):
            # The model expects encoded categorical features, but the pipeline handles the encoding.
            # We need to ensure the column order and names match the training data.
            # The pipeline handles the one-hot encoding and scaling internally.

            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[:, 1][0]

            st.subheader("Prediction Result:")
            if prediction == 1:
                st.success(f"The model predicts the customer **will** purchase the package.")
            else:
                st.info(f"The model predicts the customer **will not** purchase the package.")

            st.write(f"Prediction Probability: {prediction_proba:.2f}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.error("Please ensure your Hugging Face token is correctly set in Colab secrets and the model file exists in the repository.")
