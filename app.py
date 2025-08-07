
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("Training.csv")
    severity_df = pd.read_csv("Symptom-severity.csv")
    severity_df['Symptom'] = severity_df['Symptom'].str.lower().str.replace(' ', '_')
    label_encoder = LabelEncoder()
    df['prognosis'] = label_encoder.fit_transform(df['prognosis'])
    X = df.drop('prognosis', axis=1)
    y = df['prognosis']
    return X, y, label_encoder, severity_df

X, y, label_encoder, severity_df = load_data()
all_symptoms = X.columns.tolist()
severity_dict = dict(zip(severity_df['Symptom'], severity_df['weight']))

# Train model
@st.cache_resource
def train_model():
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X, y)
    return model

model = train_model()

# Prediction function
def predict_disease(symptom_list, k=3):
    input_vector = [0] * len(X.columns)
    for symptom in symptom_list:
        if symptom in X.columns:
            input_vector[X.columns.get_loc(symptom)] = 1

    proba = model.predict_proba([input_vector])[0]
    top_k_indices = np.argsort(proba)[-k:][::-1]
    predictions = [(label_encoder.classes_[i], proba[i]*100) for i in top_k_indices]
    return predictions

# Severity scoring
def compute_severity(symptom_list):
    score = sum([severity_dict.get(symptom, 0) for symptom in symptom_list])
    if score < 10:
        return "🟢 Low Risk", score
    elif score < 20:
        return "🟠 Moderate Risk", score
    else:
        return "🔴 High Risk", score

# App UI
st.set_page_config(page_title="Disease Predictor", layout="centered")
st.title("🧠 Disease Prediction from Symptoms")
st.write("Select your symptoms and get top-3 possible diseases with severity score.")

# Multi-select for symptoms
selected_symptoms = st.multiselect("Select symptoms:", options=all_symptoms)

# Predict button
if st.button("🔍 Predict Disease"):
    if selected_symptoms:
        preds = predict_disease(selected_symptoms)
        risk_level, severity_score = compute_severity(selected_symptoms)

        st.markdown("### ✅ Top 3 Predicted Diseases:")
        for disease, prob in preds:
            st.write(f"- {disease}: **{prob:.2f}%**")

        st.markdown(f"### ⚠️ Health Risk: {risk_level}")
        st.write(f"Symptom Severity Score: {severity_score}")
    else:
        st.warning("Please select at least one symptom to predict.")
