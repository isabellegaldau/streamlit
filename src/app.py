import pickle
import streamlit as st

# Load the trained model
with open("../models/random_forest_model.pkl", "rb") as file:
    model = pickle.load(file)

class_dict = {
    0: "Normal Weight",
    1: "Overweight Level I",
    2: "Overweight Level II",
    3: "Overweight Level III",
    4: "Obesity Type I",
    5: "Obesity Type II",
    6: "Obesity Type III"
}

st.title("BMI - Model prediction")

age = st.slider("Age", min_value=1, max_value=100, value=30, step=1, help="Select your age.")
gender = st.selectbox("Gender", ["Male", "Female"], help="Select your gender.")
height = st.slider("Height (cm)", min_value=0, max_value=250, value=170, step=1, help="Enter your height in centimeters.")
weight = st.slider("Weight (kg)", min_value=0, max_value=300, value=70, step=1, help="Enter your weight in kilograms.")
fcvc_options = {1: "Rarely", 2: "Sometimes", 3: "Regularly"}
fcvc = st.radio("Frequency of consumption of vegetables", options=fcvc_options, index=1, help="Select how often you eat vegetables in your meals.")
ncp = st.slider("Number of main meals", min_value=1, max_value=4, value=3, step=1, help="Select the number of main meals you have daily.")

if st.button("Predict"):
    gender_value = 1 if gender == "Female" else 0
    prediction = model.predict([[age, gender_value, height, weight, fcvc, ncp]])
    pred_class = class_dict[prediction[0]]
    st.write("Prediction:", pred_class)
