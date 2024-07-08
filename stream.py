import streamlit as st
import joblib
import numpy as np

# Load the model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Set page title
st.title("Stellar Classification Prediction")

# Create input fields for the user to enter data
alpha = st.number_input("Alpha (0.005 to 359.999)", min_value=0.005, max_value=359.999)
delta = st.number_input("Delta (-18.785 to 83.000)", min_value=-18.785, max_value=83.000)
u = st.number_input("U (15.349 to 28.690)", min_value=15.349, max_value=28.690)
r = st.number_input("R (13.772 to 25.408)", min_value=13.772, max_value=25.408)
run_id = st.number_input("Run ID (109 to 8162)", min_value=109.0, max_value=8162.0)
cam_col = st.number_input("Cam Col (1 to 6)", min_value=1.0, max_value=6.0)
field_id = st.number_input("Field ID (11 to 479.5)", min_value=11.0, max_value=479.5)
redshift = st.number_input("Redshift (-0.009 to 1.678)", min_value=-0.009, max_value=1.678)
plate = st.number_input("Plate (266 to 12547)", min_value=266.0, max_value=12547.0)
fiber_id = st.number_input("Fiber ID (1 to 1000)", min_value=1.0, max_value=1000.0)

# Predict button
if st.button("Predict"):
    # Transform input data
    X_test = scaler.transform(np.array([[alpha, delta, u, r, run_id, cam_col, field_id, redshift, plate, fiber_id]]))
    
    # Make prediction
    predictions = model.predict(X_test)
    output = predictions[0]
    
    # Display prediction
    if output == "GALAXY":
        st.success("The stellar class based on the specified spectral characteristics is a galaxy.")
    elif output == "QSO":
        st.success("The stellar class based on the specified spectral characteristics is a quasar.")
    elif output == "STAR":
        st.success("The stellar class based on the specified spectral characteristics is a star.")
