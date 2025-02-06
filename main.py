import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

data = pd.read_csv('DiseaseAndSymptoms.csv')
df = pd.DataFrame(data)
data.fillna('None', inplace=True)
symptom_columns = [col for col in data.columns if 'Symptom' in col]
label_encoders = {}
for column in symptom_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

X = data[symptom_columns].values
y = data['Disease'].values
le_disease = LabelEncoder()
y_encoded = le_disease.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(128, input_dim=len(symptom_columns), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(le_disease.classes_), activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=16,
          validation_data=(X_test, y_test))


def predict_disease():
    print("Enter symptoms as integers corresponding to the following list:")
    for i, symptom in enumerate(label_encoders[symptom_columns[0]].classes_):
        print(f"{i}: {symptom}")

    user_symptoms = []
    symNumbers = int(input('How many symptoms are you facing?'))
    for i in range(0, symNumbers):
        symptom_number = int(input(f"Enter Symptom {i+1}: "))
        user_symptoms.append(symptom_number)

    while len(user_symptoms) < len(symptom_columns):
        user_symptoms.append(0)

    prediction = model.predict([user_symptoms])
    predicted_disease = le_disease.inverse_transform([prediction.argmax()])

    print(f"Predicted Disease: {predicted_disease[0]}")


predict_disease()
