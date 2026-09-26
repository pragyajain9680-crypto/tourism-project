
import streamlit as st
import pandas as pd
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Wellness Tourism Prediction",
    page_icon="✈️",
    layout="wide"
)

# Load the trained model
model_path = os.path.join(
    os.path.dirname(__file__),
    "tourism_model.pkl"
)

model = joblib.load(model_path)

# Application title
st.title("Wellness Tourism Package Prediction")

st.write(
    "Enter the customer details below to predict "
    "whether the customer is likely to purchase "
    "the Wellness Tourism Package."
)

# Customer details
st.header("Customer Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    city_tier = st.selectbox(
        "City Tier",
        [1, 2, 3]
    )

    occupation = st.text_input(
        "Occupation",
        value="Salaried"
    )

    marital_status = st.selectbox(
        "Marital Status",
        ["Single", "Married", "Divorced"]
    )

    monthly_income = st.number_input(
        "Monthly Income",
        min_value=0.0,
        value=50000.0
    )

    designation = st.text_input(
        "Designation",
        value="Executive"
    )

with col2:
    type_of_contact = st.selectbox(
        "Type of Contact",
        ["Company Invited", "Self Inquiry"]
    )

    number_of_persons = st.number_input(
        "Number of Persons Visiting",
        min_value=1,
        value=2
    )

    property_star = st.number_input(
        "Preferred Property Star Rating",
        min_value=1,
        max_value=5,
        value=3
    )

    number_of_trips = st.number_input(
        "Number of Trips Annually",
        min_value=0,
        value=2
    )

    passport = st.selectbox(
        "Valid Passport",
        [0, 1]
    )

    own_car = st.selectbox(
        "Owns a Car",
        [0, 1]
    )

    number_of_children = st.number_input(
        "Number of Children Visiting",
        min_value=0,
        value=0
    )

# Customer interaction details
st.header("Customer Interaction")

col3, col4 = st.columns(2)

with col3:
    product_pitched = st.text_input(
        "Product Pitched",
        value="Basic"
    )

    pitch_satisfaction = st.number_input(
        "Pitch Satisfaction Score",
        min_value=0.0,
        value=3.0
    )

with col4:
    number_of_followups = st.number_input(
        "Number of Follow-ups",
        min_value=0,
        value=1
    )

    duration_of_pitch = st.number_input(
        "Duration of Pitch",
        min_value=0.0,
        value=10.0
    )

# Predict
if st.button("Predict Purchase", type="primary"):

    input_data = pd.DataFrame([{
        "Age": age,
        "TypeofContact": type_of_contact,
        "CityTier": city_tier,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": number_of_persons,
        "PreferredPropertyStar": property_star,
        "MaritalStatus": marital_status,
        "NumberOfTrips": number_of_trips,
        "Passport": passport,
        "OwnCar": own_car,
        "NumberOfChildrenVisiting": number_of_children,
        "Designation": designation,
        "MonthlyIncome": monthly_income,
        "PitchSatisfactionScore": pitch_satisfaction,
        "ProductPitched": product_pitched,
        "NumberOfFollowups": number_of_followups,
        "DurationOfPitch": duration_of_pitch
    }])

    try:
        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.success(
                "Prediction: The customer is likely "
                "to purchase the Wellness Tourism Package."
            )
        else:
            st.info(
                "Prediction: The customer is unlikely "
                "to purchase the Wellness Tourism Package."
            )

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            st.write(
                f"Estimated purchase probability: "
                f"{probabilities[1]:.2%}"
            )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
