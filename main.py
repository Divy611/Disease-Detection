import os
import pickle
import numpy as np
import pandas as pd
from keras.optimizers import Adam
from keras.models import Sequential
from sklearn.utils import class_weight
from keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, LearningRateScheduler

data = pd.read_csv('DiseaseAndSymptoms.csv')
data.fillna('None', inplace=True)

symptom_columns = [col for col in data.columns if 'Symptom' in col]
label_encoders = {}


for column in symptom_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

le_disease = LabelEncoder()
y = le_disease.fit_transform(data['Disease'].values)

# Split dataset
X = pd.get_dummies(data[symptom_columns])
X_train, X_test, y_train, y_test = train_test_split(
    X.values, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(512, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dense(len(le_disease.classes_), activation='softmax'))


def lr_schedule(epoch, lr):
    if epoch > 10:
        lr = lr * 0.8
    return lr


early_stopping = EarlyStopping(
    monitor='val_loss', patience=8, restore_best_weights=True)
lr_scheduler = LearningRateScheduler(lr_schedule)

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(learning_rate=0.0005), metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(
    X_test, y_test), callbacks=[early_stopping, lr_scheduler])

model.save('disease_prediction_model.h5')

pickle_folder = 'pickle_files'
if not os.path.exists(pickle_folder):
    os.makedirs(pickle_folder)

with open(os.path.join(pickle_folder, 'le_disease.pkl'), 'wb') as f:
    pickle.dump(le_disease, f)


for i, column in enumerate(symptom_columns):
    with open(os.path.join(pickle_folder, f'label_encoder_{column}.pkl'), 'wb') as f:
        pickle.dump(label_encoders[column], f)
