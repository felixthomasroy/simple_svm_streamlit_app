import streamlit as st
from svm import model, scaler


# -----------------------------
# Page settings
# -----------------------------
st.set_page_config(
    page_title="SVM Classification",
    page_icon="🤖",
    layout="centered"
)


# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>

    /* Main background */
    .stApp {
    background-color: #f3e8ff; 
   }

    /* Main title */
    h1 {
        color: #1f4e79;
        text-align: center;
    }

    /* Normal text */
    p {
        color: #333333;
    }

    /* Input labels */
    label {
        color: #1f4e79 !important;
        font-weight: bold;
    }

    /* Predict button */
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 25px;
        font-size: 16px;
        font-weight: bold;
    }

    /* Button hover */
    .stButton > button:hover {
        background-color: #155a8a;
        color: white;
    }

</style>
""", unsafe_allow_html=True)


# -----------------------------
# App title
# -----------------------------
st.title("SVM Classification App")

st.write("Predict whether a person will purchase a car.")


# -----------------------------
# User inputs
# -----------------------------
age = st.number_input(
    "Enter Age",
    min_value=18,
    max_value=100,
    value=30
)

salary = st.number_input(
    "Enter Salary",
    min_value=10000,
    max_value=200000,
    value=50000
)


# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):

    new_person = [[age, salary]]

    # Scale the input
    new_person_scaled = scaler.transform(new_person)

    # Make prediction
    prediction = model.predict(new_person_scaled)

    if prediction[0] == 1:
        st.success("Prediction: most prolly purchase")
    else:
        st.error("Prediction: won't purchase")