import numpy as np
import pandas as pd
import streamlit as st
from keras.models import load_model
import os
import pickle

st.set_page_config(page_title="Disease Prediction App", page_icon="🩺")

st.markdown('''
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        
        body {
            font-family: 'Inter', sans-serif;
            background-color: #f9fafb;
            color: #111827;
        }
        
        .block-container {
            padding: 2rem;
            background-color: #ffffff;
            box-shadow: 0px 10px 15px rgba(0, 0, 0, 0.1);
            border-radius: 12px;
            max-width: 800px;
            margin: 2rem auto;
        }
        
        h1, h2, h3 {
            color: #1f2937;
        }
        
        .stButton>button {
            background-color: #2563eb;
            color: white;
            padding: 0.75rem 1.5rem;
            border-radius: 0.375rem;
            font-weight: 600;
        }

        .stButton>button:hover {
            background-color: #1d4ed8;
        }
        
        .prediction-result {
            background-color: #f0fdf4;
            padding: 1rem;
            border-radius: 0.375rem;
            color: #065f46;
            font-weight: 600;
            margin-bottom: 1.5rem;
        }

        .precautions {
            background-color: #fef3c7;
            padding: 1rem;
            border-radius: 0.375rem;
            color: #92400e;
            margin-top: 1rem;
        }
    </style>
''', unsafe_allow_html=True)

st.title("🩺 Disease Prediction App")
model = load_model('disease_prediction_model.h5')

pickle_folder = 'pickle_files'

disease_encoder_path = os.path.join(pickle_folder, 'le_disease.pkl')
with open(disease_encoder_path, 'rb') as f:
    le_disease = pickle.load(f)

symptom_columns = ['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4', 'Symptom_5',
                   'Symptom_6', 'Symptom_7', 'Symptom_8', 'Symptom_9', 'Symptom_10',
                   'Symptom_11', 'Symptom_12', 'Symptom_13', 'Symptom_14', 'Symptom_15',
                   'Symptom_16', 'Symptom_17']

label_encoders = {}
for column in symptom_columns:
    label_encoder_path = os.path.join(
        pickle_folder, f'label_encoder_{column}.pkl')
    with open(label_encoder_path, 'rb') as f:
        label_encoders[column] = pickle.load(f)

precautions_data = pd.read_csv('Disease precaution.csv')

sym_count = st.number_input(
    'How many symptoms are you experiencing?', min_value=1, max_value=17, step=1)

user_symptoms = []
for i in range(sym_count):
    symptom_choice = st.selectbox(f"Select Symptom {i+1}:",
                                  options=label_encoders[symptom_columns[0]].classes_,
                                  key=f'symptom_{i}')
    encoded_symptom = label_encoders[symptom_columns[0]].transform([symptom_choice])[
        0]
    user_symptoms.append(encoded_symptom)

while len(user_symptoms) < 17:
    user_symptoms.append(0)
user_symptoms_np = np.array(user_symptoms).reshape(1, -1)

if st.button("Predict"):
    prediction = model.predict(user_symptoms_np)
    predicted_disease = le_disease.inverse_transform([prediction.argmax()])[0]

    st.write(
        f"<div class='prediction-result'>Predicted Disease: {predicted_disease}</div>", unsafe_allow_html=True)

    precautions = precautions_data[precautions_data['Disease'] == predicted_disease][[
        'Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values[0]

    st.write("<div class='precautions'><strong>Precautions to take:</strong></div>",
             unsafe_allow_html=True)
    for precaution in precautions:
        if precaution and precaution != 'None':
            st.write(
                f"<div class='precautions'>- {precaution}</div>", unsafe_allow_html=True)
