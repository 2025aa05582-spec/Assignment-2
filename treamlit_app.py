import streamlit as st
import pandas as pd
import numpy as np

from models import train_and_evaluate_models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

st.title("Automatic Classification Model Trainer")

# ======================
# Upload Dataset
# ======================

uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # ======================
    # Target Column Selection
    # ======================

    target_column = st.selectbox(
        "Select Target Column for Classification",
        df.columns
    )

    if target_column:

        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # ======================
        # Check if target is valid for classification
        # ======================

        if y.nunique() > 20 and y.dtype != "object":
            st.error("Selected target looks continuous. This appears to be a regression problem.")
            st.stop()

        # ======================
        # Encode Categorical Features
        # ======================

        X = pd.get_dummies(X, drop_first=True)

        # Encode target if categorical
        if y.dtype == "object":
            le = LabelEncoder()
            y = le.fit_transform(y)

        # ======================
        # Train-Test Split
        # ======================

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # ======================
        # Scaling (only numeric)
        # ======================

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model_options = [
                "All Models",
                "Logistic Regression",
                "Decision Tree",
                "KNN",
                "Naive Bayes",
                "Random Forest",
                "XGBoost"
            ]

        model_sel = st.selectbox(
            "Select Model to Train",
            model_options
        )

        # ======================
        # Train Models
        # ======================

        if st.button("Train All Classification Models"):

            results_df, model_train, pred = train_and_evaluate_models(
                X_train,
                X_test,
                y_train,
                y_test
            )

            st.success("Training Completed!")

            st.subheader("Model Performance Comparison")
            st.dataframe(results_df)

            # Highlight Best Model
            best_model = results_df["Accuracy"].astype(float).idxmax()
            st.success(f"Best Model (by Accuracy): {best_model}")

        if st.button("Train Model(s)"): 
            results_df, pred = train_and_evaluate_models(
                            X_train,
                            X_test,
                            y_train,
                            y_test,
                            model_sel)
            
            st.success("Training Completed!")

            st.subheader(f"{model_sel} Performance")
            if model_sel != "All Models":
                model_name = model_sel
            y_pred = pred[model_name]

            st.subheader(f"Metrics for {model_name}")
            st.dataframe(results_df)

            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)

            st.subheader("Confusion Matrix")

            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(f"{model_name} Confusion Matrix")

            st.pyplot(fig)
            st.dataframe(results_df)

            
            
   
