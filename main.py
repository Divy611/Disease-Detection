import os
import pickle
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

data = pd.read_csv('DiseaseAndSymptoms.csv')
data.fillna('None', inplace=True)

disease_counts = data['Disease'].value_counts()
print(f"Number of diseases: {len(disease_counts)}")
print(f"Most common diseases: {disease_counts.head(5)}")
print(f"Least common diseases: {disease_counts.tail(5)}")

symptom_columns = [col for col in data.columns if 'Symptom' in col]


def create_symptom_features(data, symptom_columns):
    all_symptoms = set()
    for col in symptom_columns:
        all_symptoms.update(data[col].unique())
    if 'None' in all_symptoms:
        all_symptoms.remove('None')
    all_symptoms = sorted(list(all_symptoms))
    symptom_to_idx = {symptom: i for i, symptom in enumerate(all_symptoms)}

    X = np.zeros((len(data), len(all_symptoms)))
    for i, row in data.iterrows():
        for col in symptom_columns:
            symptom = row[col]
            if symptom != 'None':
                X[i, symptom_to_idx[symptom]] = 1
    return X, all_symptoms


X, symptom_names = create_symptom_features(data, symptom_columns)

le_disease = LabelEncoder()
y = le_disease.fit_transform(data['Disease'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

models = {
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=20,
                                            min_samples_split=5, random_state=42,
                                            class_weight='balanced'),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=150, learning_rate=0.1,
                                                    max_depth=7, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=150, learning_rate=0.1, max_depth=7,
                             random_state=42, use_label_encoder=False, eval_metric='mlogloss')
}

results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f"{name} Accuracy: {accuracy:.4f}")
    print(f"{name} Classification Report:")
    print(classification_report(y_test, y_pred))

best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
print(
    f"\nBest model: {best_model_name} with accuracy {results[best_model_name]:.4f}")

if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("\nTop 20 most important symptoms:")
    for i in range(min(20, len(symptom_names))):
        print(f"{symptom_names[indices[i]]}: {importances[indices[i]]:.4f}")


def predict_disease(symptoms, model, symptom_names, le_disease):
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

    if not_found_symptoms:
        print(
            f"Warning: These symptoms were not found in the training data: {not_found_symptoms}")

    if not found_symptoms:
        print("No valid symptoms provided.")
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


test_cases = [
    ["bladder_discomfort", "burning_micturition"],
    ["back_pain", "cramps", "fatigue"]
]

print("\nTesting problematic cases:")
for symptoms in test_cases:
    print(f"\nSymptoms: {symptoms}")
    predictions = predict_disease(
        symptoms, best_model, symptom_names, le_disease)
    if predictions:
        for disease, prob in predictions:
            print(f"- {disease}: {prob:.4f} ({prob*100:.1f}%)")

model_folder = 'improved_model'
if not os.path.exists(model_folder):
    os.makedirs(model_folder)

with open(os.path.join(model_folder, 'disease_prediction_model.pkl'), 'wb') as f:
    pickle.dump(best_model, f)

with open(os.path.join(model_folder, 'le_disease.pkl'), 'wb') as f:
    pickle.dump(le_disease, f)

with open(os.path.join(model_folder, 'symptom_names.pkl'), 'wb') as f:
    pickle.dump(symptom_names, f)

print(f"\nModel and necessary files saved to '{model_folder}' directory")


def predict_from_symptoms(symptoms, model_folder='improved_model'):
    with open(os.path.join(model_folder, 'disease_prediction_model.pkl'), 'rb') as f:
        model = pickle.load(f)

    with open(os.path.join(model_folder, 'le_disease.pkl'), 'rb') as f:
        le_disease = pickle.load(f)

    with open(os.path.join(model_folder, 'symptom_names.pkl'), 'rb') as f:
        symptom_names = pickle.load(f)
    return predict_disease(symptoms, model, symptom_names, le_disease)


def get_prediction_for_streamlit(symptoms, model_folder='improved_model'):
    predictions = predict_from_symptoms(symptoms, model_folder)
    if not predictions:
        return []

    result = []
    for disease, probability in predictions:
        result.append({
            'disease': disease,
            'probability': probability,
            'percentage': f"{probability*100:.1f}%"
        })

    return result
