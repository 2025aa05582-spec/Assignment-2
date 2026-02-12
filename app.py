import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from models import train_and_evaluate_models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix

st.title("Classification Model Trainer")

# ======================
# Upload Dataset
# ======================

uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # ======================
    # Select Target Column
    # ======================

    target_column = st.selectbox("Select Target Column", df.columns)

    # ======================
    # Select Model
    # ======================

    model_options = [
        "All Models",
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]

    selected_model = st.selectbox("Select Model to Train", model_options)

    if st.button("Train Model(s)"):

        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # ======================
        # Automatic Encoding
        # ======================

        X = pd.get_dummies(X, drop_first=True)

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
        # Scaling
        # ======================

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # ======================
        # Train Models
        # ======================

        results_df, predictions = train_and_evaluate_models(
            selected_model,
            X_train,
            X_test,
            y_train,
            y_test,
        )

        st.success("Training Completed!")

        st.subheader("Model Performance Metrics")
        st.dataframe(results_df)

        # ======================
        # Display Sample Predictions (First 10)
        # ======================

        st.subheader("Sample Predictions (First 10 Rows)")

        # Convert y_test to Series (if numpy array)
        y_test_series = pd.Series(y_test).reset_index(drop=True)

        if selected_model == "All Models":

            for model_name, y_pred in predictions.items():

                st.markdown(f"### {model_name}")

                y_pred_series = pd.Series(y_pred).reset_index(drop=True)

                prediction_df = pd.DataFrame({
                    "Actual": y_test_series[:10],
                    "Predicted": y_pred_series[:10]
                })

                st.dataframe(prediction_df)

        else:

            y_pred = predictions[selected_model]
            y_pred_series = pd.Series(y_pred).reset_index(drop=True)

            prediction_df = pd.DataFrame({
                "Actual": y_test_series[:10],
                "Predicted": y_pred_series[:10]
            })

            st.dataframe(prediction_df)


        # ======================
        # Confusion Matrix (Only for Single Model)
        # ======================

        if selected_model != "All Models":

            y_pred = predictions[selected_model]
            cm = confusion_matrix(y_test, y_pred)

            st.subheader("Confusion Matrix")

            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(f"{selected_model} Confusion Matrix")

            st.pyplot(fig)
