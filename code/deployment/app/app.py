import requests
import streamlit as st


API_URL = "http://api:8000/predict"

st.title("Palmer Penguins Classification")

bill_length = st.number_input("Bill length (mm)", value=39.1)
bill_depth = st.number_input("Bill depth (mm)", value=18.7)
flipper_length = st.number_input("Flipper length (mm)", value=181.0)
body_mass = st.number_input("Body mass (g)", value=3750.0)

island = st.selectbox(
    "Island",
    ["Biscoe", "Dream", "Torgersen"],
)

sex = st.selectbox(
    "Sex",
    ["male", "female"],
)

if st.button("Predict"):
    data = {
        "bill_length_mm": bill_length,
        "bill_depth_mm": bill_depth,
        "flipper_length_mm": flipper_length,
        "body_mass_g": body_mass,
        "island": island,
        "sex": sex,
    }

    response = requests.post(
        API_URL,
        json=data,
        timeout=10,
    )

    if response.ok:
        result = response.json()

        st.success(
            f"Predicted species: {result['prediction']}"
        )

        st.write("Probabilities:")

        for species, probability in result["probabilities"].items():
            st.write(
                f"{species}: {probability:.2%}"
            )
    else:
        st.error("Prediction request failed.")