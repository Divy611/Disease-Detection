import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from keras.models import load_model

model = load_model('disease_prediction_model.h5')
pickle_folder = 'pickle_files'

disease_encoder_path = os.path.join(pickle_folder, 'le_disease.pkl')
if not os.path.exists(disease_encoder_path):
    st.error("Disease label encoder not found!")
    st.stop()
with open(disease_encoder_path, 'rb') as f:
    le_disease = pickle.load(f)

symptom_columns = ['Symptom_1', 'Symptom_2',
                   'Symptom_3', 'Symptom_4', 'Symptom_5']
label_encoders = {}
for column in symptom_columns:
    label_encoder_path = os.path.join(
        pickle_folder, f'label_encoder_{column}.pkl')
    if not os.path.exists(label_encoder_path):
        st.error(f"Label encoder for {column} not found!")
        st.stop()
    with open(label_encoder_path, 'rb') as f:
        label_encoders[column] = pickle.load(f)

st.title("Disease Prediction App")

sym_count = st.number_input(
    'How many symptoms are you facing?', min_value=1, step=1)

user_symptoms = []
for i in range(sym_count):
    symptom_choice = st.selectbox(f"Enter symptom {i+1}:",
                                  options=label_encoders[symptom_columns[0]].classes_,
                                  key=f'symptom_{i}')

    encoded_symptom = label_encoders[symptom_columns[0]].transform([symptom_choice])[
        0]
    user_symptoms.append(encoded_symptom)

while len(user_symptoms) < len(symptom_columns):
    user_symptoms.append(0)
if st.button("Predict"):
    prediction = model.predict([user_symptoms])
    predicted_disease = le_disease.inverse_transform([prediction.argmax()])
    st.write(f"Predicted Disease: {predicted_disease[0]}")
