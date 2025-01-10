import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import numpy as np

# Streamlit app
st.title("🩺Healthcare Finder - Diabetes Detection")
option = st.sidebar.radio("Choose an action:", ["Train a Model", "Make Predictions"])

# Function to preprocess diabetes data
def preprocess_diabetes_data(data):
    return data

# Function to train the model
def train_model(data, model_choice):
    try:
        X = data.drop(columns=["Outcome"])
        y = data["Outcome"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Choose model
        if model_choice == "Logistic Regression":
            model = LogisticRegression(max_iter=200)
        elif model_choice == "Random Forest":
            model = RandomForestClassifier()
        elif model_choice == "SVM":
            model = SVC(probability=True)
        else:
            raise ValueError("Invalid model selected")

        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        return model, accuracy, cm
    except Exception as e:
        raise RuntimeError(f"Model training failed: {e}")

# Function for predictions
def predict_diabetes(model, features):
    prediction = model.predict([features])
    probability = model.predict_proba([features])
    return ("Positive" if prediction == 1 else "Negative", probability[0][1])

if option == "Train a Model":
    st.sidebar.header("Train a Model")
    
    # Generate synthetic data for training
    glucose = np.random.randint(70, 180, size=100)
    bp = np.random.randint(60, 90, size=100)
    skin = np.random.randint(15, 40, size=100)
    insulin = np.random.randint(15, 200, size=100)
    bmi = np.random.uniform(18.0, 40.0, size=100)
    dpf = np.random.uniform(0.1, 2.5, size=100)
    age = np.random.randint(20, 80, size=100)
    outcome = np.random.randint(0, 2, size=100)
    
    train_data = pd.DataFrame({
        "Glucose": glucose,
        "BloodPressure": bp,
        "SkinThickness": skin,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age,
        "Outcome": outcome
    })
    
    st.write("Sample training data:")
    st.dataframe(train_data.head())

    # Model selection
    model_choice = st.sidebar.selectbox("Select a model:", ["Logistic Regression", "Random Forest", "SVM"])
    
    if st.button("Train Model"):
        try:
            model, accuracy, cm = train_model(train_data, model_choice)
            st.session_state["model"] = model
            st.success(f"Model trained successfully with accuracy: {accuracy*100:.2f}%")
            st.write("Confusion Matrix:")
            st.write(cm)
        except Exception as e:
            st.error(f"Error: {e}")

elif option == "Make Predictions":
    if "model" not in st.session_state:
        st.error("No model trained yet. Please train a model first.")
    else:
        st.sidebar.header("Patient Data Input")
        glucose = st.sidebar.slider("Glucose Level", 0, 200, 100)
        bp = st.sidebar.slider("Blood Pressure", 0, 200, 70)
        skin = st.sidebar.slider("Skin Thickness", 0, 100, 20)
        insulin = st.sidebar.slider("Insulin Level", 0, 900, 100)
        bmi = st.sidebar.slider("BMI", 0.0, 60.0, 25.0)
        dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
        age = st.sidebar.slider("Age", 18, 120, 30)

        patient_features = [glucose, bp, skin, insulin, bmi, dpf, age]

        if st.sidebar.button("Predict"):
            prediction, probability = predict_diabetes(st.session_state["model"], patient_features)
            st.success(f"Prediction: The patient is **{prediction}** for diabetes.")
            st.info(f"Prediction Confidence: {probability*100:.2f}%")













































