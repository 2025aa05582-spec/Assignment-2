import streamlit as st
import pandas as pd
import joblib

from models import train_and_evaluate_models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# ======================
# Upload Dataset
# ======================

uploaded_file = st.file_uploader("Upload Dataset", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview")
    st.dataframe(df.head())

    # Features & Target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )



    # Encode categorical columns automatically
    X = pd.get_dummies(X)

    # If target is categorical, encode it too
    if y.dtype == "object":
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
)

    # ======================
    # Scaling
    # ======================
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler
    joblib.dump(scaler, "model/scaler.pkl")

    # ======================
    # Train Models
    # ======================
    if st.button("Train Models"):

        results_df = train_and_evaluate_models(
            X_train_scaled,
            X_test_scaled,
            y_train,
            y_test
        )

        st.success("Models Trained Successfully!")

        st.write("Model Performance Comparison")
        st.dataframe(results_df)
