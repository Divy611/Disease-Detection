import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Disease Prediction App",
                   page_icon="🩺", layout="wide")
st.markdown('''
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        
        body {
            color: #f3f4f6;
            background-color: #111827;
            font-family: 'Inter', sans-serif;
        }
        
        .block-container {
            padding: 2rem;
            max-width: 1200px;
            margin: 2rem auto;
            border-radius: 12px;
            background-color: #1f2937;
            box-shadow: 0px 10px 15px rgba(0, 0, 0, 0.2);
        }
        
        h1, h2, h3 {color: #f9fafb;}
        
        .stButton>button {
            width: 100%;
            color: white;
            font-weight: 600;
            padding: 0.75rem 1.5rem;
            border-radius: 0.375rem;
            background-color: #3b82f6;
        }
        .stButton>button:hover {background-color: #2563eb;}
        
        .prediction-card {
            padding: 1.5rem;
            margin-top: 1rem;
            border-radius: 0.75rem;
            background-color: #374151;
            border-left: 6px solid #3b82f6;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
        }
        
        .prediction-result-high {border-left: 4px solid #10b981;}
        .prediction-result-medium {border-left: 4px solid #f59e0b;}
        .prediction-result-low {border-left: 4px solid #ef4444;}

        .precautions {
            padding: 1rem;
            color: #fbbf24;
            margin-top: 1rem;
            border-radius: 0.375rem;
            background-color: #4b5563;
        }

        .selected-symptoms {
            padding: 1rem;
            color: #bfdbfe;
            margin-bottom: 1rem;
            border-radius: 0.375rem;
            background-color: #1e3a8a;
        }

        table {width: 100%;border-collapse: collapse;}
        th, td {
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid #4b5563;
        }

        th {color: #e5e7eb;background-color: #374151;}
        td {color: #d1d5db;background-color: #1f2937;}
        .progress-container {
            display: flex;
            align-items: center;
            margin-bottom: 0.5rem;
        }
        
        .progress-label {
            color: #e5e7eb;
            min-width: 120px;
            margin-right: 10px;
        }
        
        .progress-bar {
            height: 12px;
            flex-grow: 1;
            overflow: hidden;
            border-radius: 6px;
            position: relative;
            background-color: #4b5563;
        }
        
        .progress-fill {
            height: 100%;
            border-radius: 6px;
        }
        
        .progress-text {
            color: #e5e7eb;
            min-width: 60px;
            font-weight: 600;
            margin-left: 10px;
        }
        
        .css-1dp5vir {
            color: #e5e7eb;
            border: 1px solid #374151;
            background-color: #1f2937;
        }
    
        .css-1vq4p4l, .css-1d0tddh {color: #e5e7eb;}
        .css-1adrfps {color: #e5e7eb;background-color: #374151;}
        .stMultiSelect > div > div {color: #e5e7eb;background-color: #374151;}
        .stMultiSelect > div[data-baseweb="select"] > div {color: #e5e7eb;background-color: #4b5563;}
        .stSelectbox > div > div > div {color: #e5e7eb;background-color: #374151;}
        .stNumberInput > div > div > div {color: #e5e7eb;background-color: #374151;}
        .stTextInput > div > div > input {color: #e5e7eb;background-color: #374151;}
        section[data-testid="stSidebar"] {
            background-color: #1f2937;
            border-right: 1px solid #374151;
        }
        
        .stAlert {color: #e5e7eb;background-color: #374151;}
        .stAlert a {color: #60a5fa;}
        h4 {color: #e5e7eb;margin-top: 0;}
    </style>
''', unsafe_allow_html=True)

st.title("🩺 Advanced Disease Prediction System")
st.markdown(
    "Please select the symptoms you're experiencing for an accurate prediction.")

model_folder = 'improved_model'
if os.path.exists(os.path.join(model_folder, 'disease_prediction_model.pkl')):
    with open(os.path.join(model_folder, 'disease_prediction_model.pkl'), 'rb') as f:
        model = pickle.load(f)

    with open(os.path.join(model_folder, 'le_disease.pkl'), 'rb') as f:
        le_disease = pickle.load(f)

    with open(os.path.join(model_folder, 'symptom_names.pkl'), 'rb') as f:
        symptom_names = pickle.load(f)
    improved_model = True
else:
    model = None
    try:
        from keras.models import load_model
        model = load_model('disease_prediction_model.h5')
    except:
        st.error("Legacy model file not found. Please run the training script first.")
        st.stop()

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

    improved_model = False

try:
    precautions_data = pd.read_csv('Disease precaution.csv')
except:
    st.warning("Precaution data not found. Precautions will not be displayed.")
    precautions_data = None


def predict_with_improved_model(symptoms):
    X_new = np.zeros(len(symptom_names))

    found_symptoms = []
    not_found_symptoms = []

    for symptom in symptoms:
        if symptom in symptom_names:
            idx = symptom_names.index(symptom)
            X_new[idx] = 1
            found_symptoms.append(symptom)
        else:
            not_found_symptoms.append(symptom)

    if not found_symptoms:
        return None

    X_new = X_new.reshape(1, -1)
    y_pred = model.predict(X_new)
    probabilities = model.predict_proba(X_new)[0]
    top_indices = np.argsort(probabilities)[::-1][:5]

    results = []
    for idx in top_indices:
        disease = le_disease.inverse_transform([idx])[0]
        prob = probabilities[idx]
        results.append((disease, prob))

    return results


st.markdown("## Select Your Symptoms")
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Symptom Selection")
    if improved_model:
        available_symptoms = sorted(symptom_names)
        selected_symptoms = st.multiselect(
            "Select all symptoms you're experiencing:",
            options=available_symptoms
        )

        if len(selected_symptoms) == 0:
            st.info("Please select at least one symptom")
    else:
        sym_count = st.number_input(
            'How many symptoms are you experiencing?',
            min_value=1, max_value=17, step=1
        )

        user_symptoms = []
        for i in range(sym_count):
            symptom_choice = st.selectbox(
                f"Select Symptom {i+1}:",
                options=label_encoders[symptom_columns[0]].classes_,
                key=f'symptom_{i}'
            )
            encoded_symptom = label_encoders[symptom_columns[0]].transform([symptom_choice])[
                0]
            user_symptoms.append(encoded_symptom)

        while len(user_symptoms) < 17:
            user_symptoms.append(0)
        user_symptoms_np = np.array(user_symptoms).reshape(1, -1)
    predict_button = st.button("Predict Disease", use_container_width=True)

with col2:
    st.subheader("Results")

    if improved_model and predict_button and len(selected_symptoms) > 0:
        st.markdown("<div class='selected-symptoms'><strong>Selected Symptoms:</strong><br>" +
                    ", ".join(selected_symptoms) + "</div>", unsafe_allow_html=True)

        predictions = predict_with_improved_model(selected_symptoms)

        if predictions:
            st.markdown("<h3>Possible Diagnoses</h3>", unsafe_allow_html=True)
            for i, (disease, probability) in enumerate(predictions):
                if probability >= 0.5:
                    color_class = "prediction-result-high"
                elif probability >= 0.25:
                    color_class = "prediction-result-medium"
                else:
                    color_class = "prediction-result-low"

                percentage = f"{probability*100:.1f}%"

                st.markdown(f"""
                <div class='prediction-card {color_class}'>
                    <h4>{disease}</h4>
                    <div class='progress-container'>
                        <div class='progress-label'>Chances:</div>
                        <div class='progress-bar'>
                            <div class='progress-fill' style='width: {percentage}; background-color: {"#10b981" if probability >= 0.5 else "#f59e0b" if probability >= 0.25 else "#ef4444"};'></div>
                        </div>
                        <div class='progress-text'>{percentage}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if i == 0 and precautions_data is not None:
                    precautions = precautions_data[precautions_data['Disease'] == disease][[
                        'Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]

                    if not precautions.empty:
                        st.markdown("<h3>Recommended Precautions</h3>",
                                    unsafe_allow_html=True)
                        precautions = precautions.values[0]

                        st.markdown("""
                        <table>
                            <thead>
                                <tr>
                                    <th>Precaution #</th>
                                    <th>Description</th>
                                </tr>
                            </thead>
                            <tbody>
                        """, unsafe_allow_html=True)

                        for j, precaution in enumerate(precautions, start=1):
                            if precaution and str(precaution).lower() != 'none':
                                st.markdown(f"""
                                <tr>
                                    <td>{j}</td>
                                    <td>{precaution}</td>
                                </tr>
                                """, unsafe_allow_html=True)

                        st.markdown("""
                            </tbody>
                        </table>
                        """, unsafe_allow_html=True)

    elif not improved_model and predict_button:
        prediction = model.predict(user_symptoms_np)
        predicted_disease = le_disease.inverse_transform(
            [prediction.argmax()])[0]

        st.markdown(f"""
        <div class='prediction-card prediction-result-high'>
            <h4>Predicted Disease</h4>
            <p style='font-size: 1.2rem; font-weight: 600; color: #e5e7eb;'>{predicted_disease}</p>
        </div>
        """, unsafe_allow_html=True)

        if precautions_data is not None:
            precautions = precautions_data[precautions_data['Disease'] == predicted_disease][[
                'Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]

            if not precautions.empty:
                st.markdown("<h3>Recommended Precautions</h3>",
                            unsafe_allow_html=True)
                precautions = precautions.values[0]
                st.markdown("""
                <table>
                    <thead>
                        <tr>
                            <th>Precaution #</th>
                            <th>Description</th>
                        </tr>
                    </thead>
                    <tbody>
                """, unsafe_allow_html=True)

                for i, precaution in enumerate(precautions, start=1):
                    if precaution and str(precaution).lower() != 'none':
                        st.markdown(f"""
                        <tr>
                            <td>{i}</td>
                            <td>{precaution}</td>
                        </tr>
                        """, unsafe_allow_html=True)

                st.markdown("""
                    </tbody>
                </table>
                """, unsafe_allow_html=True)
