import os
import pickle
import numpy as np
import pandas as pd
from itertools import cycle
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc, precision_recall_curve

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

if not os.path.exists('graphs'):
    os.makedirs('graphs')

models = {
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=20,
                                            min_samples_split=5, random_state=42,
                                            class_weight='balanced'),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=150, learning_rate=0.1,
                                                    max_depth=7, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=150, learning_rate=0.1, max_depth=7,
                             random_state=42, eval_metric='mlogloss',
                             num_class=len(np.unique(y)))
}


def plot_learning_curve(estimator, X, y, title):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-',
             color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-',
             color="g", label="Cross-validation score")
    plt.legend(loc="best")
    plt.savefig(f"graphs/{title.replace(' ', '_')}_learning_curve.png")
    plt.close()


def plot_multiclass_roc(model, X_test, y_test, model_name, n_classes):
    y_test_bin = label_binarize(y_test, classes=range(n_classes))
    if isinstance(model, XGBClassifier):
        xgb_model = XGBClassifier(
            n_estimators=model.n_estimators,
            learning_rate=model.learning_rate,
            max_depth=model.max_depth,
            random_state=model.random_state,
            # use_label_encoder=model.use_label_encoder,
            eval_metric=model.eval_metric,
            num_class=n_classes
        )
        classifier = OneVsRestClassifier(xgb_model)
    else:
        classifier = OneVsRestClassifier(model)

    y_score = classifier.fit(X_train, y_train).predict_proba(X_test)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        if i < y_score.shape[1]:
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    colors = cycle(['blue', 'red', 'green', 'cyan', 'magenta'])

    for i, color in zip(range(min(5, n_classes)), colors):
        if i in roc_auc:
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'ROC curve of class {i} (area = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Multi-class ROC - {model_name}')
    plt.legend(loc="lower right")
    plt.savefig(f"graphs/{model_name.replace(' ', '_')}_multiclass_roc.png")
    plt.close()


def plot_precision_recall_curve(model, X_test, y_test, model_name, n_classes):
    y_test_bin = label_binarize(y_test, classes=range(n_classes))
    if isinstance(model, XGBClassifier):
        xgb_model = XGBClassifier(
            n_estimators=model.n_estimators,
            learning_rate=model.learning_rate,
            max_depth=model.max_depth,
            random_state=model.random_state,
            # use_label_encoder=model.use_label_encoder,
            eval_metric=model.eval_metric,
            num_class=n_classes
        )
        classifier = OneVsRestClassifier(xgb_model)
    else:
        classifier = OneVsRestClassifier(model)

    y_score = classifier.fit(X_train, y_train).predict_proba(X_test)
    precision = dict()
    recall = dict()

    plt.figure(figsize=(10, 8))
    colors = cycle(['blue', 'red', 'green', 'cyan', 'magenta'])

    for i, color in zip(range(min(5, n_classes)), colors):
        if i < y_score.shape[1]:
            precision[i], recall[i], _ = precision_recall_curve(
                y_test_bin[:, i], y_score[:, i])
            plt.plot(recall[i], precision[i], color=color, lw=2,
                     label=f'Precision-Recall for class {i}')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend(loc="best")
    plt.savefig(f"graphs/{model_name.replace(' ', '_')}_precision_recall.png")
    plt.close()


def create_architecture_diagram():
    fig, ax = plt.figure(figsize=(10, 8)), plt.gca()
    ax.axis('off')

    boxes = ['Data Collection', 'Data Preprocessing', 'Feature Engineering',
             'Model Training', 'Model Evaluation', 'Prediction']

    box_positions = [(0.5, 0.85), (0.5, 0.7), (0.5, 0.55),
                     (0.5, 0.4), (0.5, 0.25), (0.5, 0.1)]

    for i, (box, pos) in enumerate(zip(boxes, box_positions)):
        rect = plt.Rectangle((pos[0]-0.15, pos[1]-0.05), 0.3, 0.1,
                             fill=True, color='skyblue', alpha=0.7)
        ax.add_patch(rect)
        ax.text(pos[0], pos[1], box, ha='center', va='center', fontsize=12)

        if i < len(boxes) - 1:
            ax.arrow(pos[0], pos[1]-0.05, 0, -0.1, head_width=0.02,
                     head_length=0.02, fc='black', ec='black')

    plt.title('Disease Prediction Model Architecture', fontsize=14)
    plt.savefig("graphs/architecture_diagram.png")
    plt.close()


n_classes = len(np.unique(y))
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

for name, model in models.items():
    print(f"\nGenerating evaluation metrics for {name}...")
    plot_learning_curve(model, X_train, y_train, f"{name} Learning Curve")

    plot_multiclass_roc(model, X_test, y_test, name, n_classes)
    plot_precision_recall_curve(model, X_test, y_test, name, n_classes)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy

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

accuracies = [results[model] for model in models.keys()]
plt.figure(figsize=(10, 6))
plt.bar(models.keys(), accuracies)
plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.savefig("graphs/model_accuracy_comparison.png")
plt.close()
create_architecture_diagram()

print("Evaluation metrics, graphs, and architecture diagram have been generated.")


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
