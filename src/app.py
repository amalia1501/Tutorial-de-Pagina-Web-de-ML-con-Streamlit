# from utils import db_connect
# engine = db_connect()



# your code here
import streamlit as st
from pickle import load


# st.title("Hello, World!")


model_path = "../models/xgbboost_nestimators-50_min_samples_leaf_1_min_samples_split_2_42.sav"
with open(model_path, "rb") as file:
    model = load(file)

class_dict = {
    "0": "Negative",
    "1": "Positive"
}

st.title("Outcome - Model prediction")

val1 = st.slider("Pregnancies", min_value = 0, max_value = 10, step = 1)
val2 = st.slider("Glucose", min_value = 0, max_value = 200, step = 1)
val3 = st.slider("BloodPressure", min_value = 0, max_value = 200, step = 1)
val4 = st.slider("SkinThickness", min_value = 0, max_value = 100, step = 1)
val5 = st.slider("Insulin", min_value = 0, max_value = 1000, step = 5)
val6 = st.slider("BMI", min_value = 0, max_value = 50, step = 1)
val7 = st.slider("DiabetesPedigreeFunction", min_value = 0.00, max_value = 1.00, step = 0.01)
val8 = st.slider("Age", min_value = 0, max_value = 100, step = 1)

if st.button("Predict"):
    prediction = str(model.predict([[val1, val2, val3, val4, val5, val6, val7, val8]])[0])
    pred_class = class_dict[prediction]
    st.write("Prediction:", pred_class)
