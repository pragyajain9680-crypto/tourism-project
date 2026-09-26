import os
import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Wellness Tourism Prediction",
    page_icon="✈️",
    layout="wide"
)

# --------------------------------------------------
# 2. Load the trained machine-learning model
# --------------------------------------------------

model_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "tourism_model.pkl"
)

@st.cache_resource
def load_model(path):
    return joblib.load(path)

try:
    if not os.path.exists(model_path):
        st.error(
            "Model file not found. Please check that "
            "tourism_model.pkl is in the repository."
        )
        st.stop()

    model = load_model(model_path)

except Exception as e:
    st.error(f"Unable to load the trained model: {e}")
    st.stop()

# --------------------------------------------------
# 3. Application title
# --------------------------------------------------

st.title("Wellness Tourism Package Prediction")

st.write(
    "Enter the customer details below to predict "
    "whether the customer is likely to purchase "
    "the Wellness Tourism Package."
)

# --------------------------------------------------
# 4. Customer details
# --------------------------------------------------

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
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    own_car = st.selectbox(
        "Owns a Car",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    number_of_children = st.number_input(
        "Number of Children Visiting",
        min_value=0,
        value=0
    )

# --------------------------------------------------
# 5. Customer interaction details
# --------------------------------------------------

st.header("Customer Interaction")

col3, col4 = st.columns(2)

with col3:

    product_pitched = st.selectbox(
        "Product Pitched",
        ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
    )

    pitch_satisfaction = st.number_input(
        "Pitch Satisfaction Score",
        min_value=0.0,
        max_value=5.0,
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

# --------------------------------------------------
# 6. Prepare customer data for prediction
# --------------------------------------------------

if st.button("Predict Purchase", type="primary"):

    customer_data = {
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
    }

    # Include the extra index column reported
    # by the earlier error.
    customer_data["Unnamed: 0"] = 0

    input_data = pd.DataFrame([customer_data])

    try:

        # Match the columns expected by the model.
        expected_columns = getattr(
            model,
            "feature_names_in_",
            None
        )

        if expected_columns is not None:

            expected_columns = list(expected_columns)

            missing_columns = [
                column
                for column in expected_columns
                if column not in input_data.columns
            ]

            if missing_columns:
                st.error(
                    "The trained model requires additional "
                    "input columns: "
                    + ", ".join(missing_columns)
                )
                st.stop()

            # Keep the exact expected column order.
            input_data = input_data[expected_columns]

        # Generate prediction.
        prediction = model.predict(input_data)[0]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.success(
                "The customer is likely to purchase "
                "the Wellness Tourism Package."
            )

        else:
            st.info(
                "The customer is unlikely to purchase "
                "the Wellness Tourism Package."
            )

        # Show probability when supported by the model.
        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(input_data)[0]

            classes = getattr(model, "classes_", None)

            if classes is not None and 1 in classes:
                positive_index = list(classes).index(1)

                purchase_probability = probabilities[
                    positive_index
                ]

                st.metric(
                    "Estimated Purchase Probability",
                    f"{purchase_probability:.2%}"
                )

            else:
                st.info(
                    "The model does not expose a probability "
                    "for the purchase class."
                )

    except Exception as e:

        st.error(
            "Prediction could not be completed. "
            "Please check the model's expected input "
            "columns and data types."
        )

        st.code(str(e))
