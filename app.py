import os
import numpy as np
import pandas as pd
import streamlit as st
from keras.models import load_model

model = load_model('disease_prediction_model.h5')
le_disease = pd.read_pickle('pickle_files/le_disease.pkl')

pickle_folder = 'pickle_files'
symptom_columns = [f'Symptom_{i+1}' for i in range(17)]
label_encoders = {}
for i in range(len(symptom_columns)):
    label_encoders[symptom_columns[i]] = pd.read_pickle(
        os.path.join(pickle_folder, f'label_encoder_{symptom_columns[i]}.pkl'))

st.title("Disease Prediction App")

st.write("Enter symptoms as integers corresponding to the following list:")
symptom_options = {}
for i, symptom in enumerate(label_encoders[symptom_columns[0]].classes_):
    symptom_options[i] = symptom
    st.write(f"{i}: {symptom}")

symNumbers = st.number_input("How many symptoms are you facing?",
                             min_value=1, max_value=len(symptom_columns), step=1)

user_symptoms = []
for i in range(symNumbers):
    symptom_input = st.number_input(
        f"Enter symptom {i+1}:", min_value=0, max_value=len(symptom_options)-1, step=1)
    user_symptoms.append(symptom_input)

if len(user_symptoms) > 0 and st.button("Predict"):
    while len(user_symptoms) < len(symptom_columns):
        user_symptoms.append(0)

    prediction = model.predict([user_symptoms])
    predicted_disease = le_disease.inverse_transform([np.argmax(prediction)])
    st.write(f"Predicted Disease: {predicted_disease[0]}")
