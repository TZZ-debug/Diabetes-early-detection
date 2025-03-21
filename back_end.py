import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import streamlit as st
import pickle
from fpdf import FPDF
import base64


# Data loading and Exploratory Data Analysis (EDA) module
def load_and_explore_data(file_path):
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"The dataset file '{file_path}' was not found. Please check the file name and path.")
        return None

    # Manually define mapping relationships
    gender_mapping = {'F': 0, 'M': 1}
    class_mapping = {'N': 0, 'P': 1, 'Y': 2}

    # Encode the Gender and Class columns
    data['Gender'] = data['Gender'].str.strip().map(gender_mapping)
    data['CLASS'] = data['CLASS'].str.strip().map(class_mapping)

    # View basic information of the data
    #st.write("\nData Description:")
    #st.write(data.describe())

    # Check for missing values
    #st.write("\nMissing Values:")
    #st.write(data.isnull().sum())

    # Visualize data distribution
    '''
    num_columns = len(data.columns[:-1])
    num_rows = math.ceil(num_columns / 3)
    plt.figure(figsize=(12, 4 * num_rows))
    for i, column in enumerate(data.columns[:-1], 1):
        plt.subplot(num_rows, 3, i)
        sns.histplot(data[column], kde=True, bins=30)
        plt.title(f'Distribution of {column}')
    plt.tight_layout()
    st.pyplot()

    # Correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    st.pyplot()
    '''

    return data


# Data preprocessing module
def preprocess_data(data):
    if data is None:
        return None, None

    # Separate features and labels. Assume CLASS is the target variable
    X = data.drop(columns=['CLASS']).drop(columns=['ID']).drop(columns=['No_Pation'])
    y = data['CLASS']

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save the preprocessing object
    with open('preprocessor.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test


# Model development and evaluation module
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    if X_train is None or X_test is None or y_train is None or y_test is None:
        return None, None

    # Define models
    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }

    # Train and evaluate
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        results[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average='macro'),
            "Recall": recall_score(y_test, y_pred, average='macro'),
            "F1-Score": f1_score(y_test, y_pred, average='macro'),
            "AUC-ROC": roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
        }

    # Display results
    results_df = pd.DataFrame(results).T
    #st.write("\nModel Evaluation Results:")
    #st.write(results_df)

    return results_df


# Ensemble learning module
def create_and_evaluate_voting_clf(X_train, X_test, y_train, y_test):
    if X_train is None or X_test is None or y_train is None or y_test is None:
        return None

    # Create an ensemble model
    voting_clf = VotingClassifier(estimators=[
        ('lr', LogisticRegression()),
        ('rf', RandomForestClassifier(random_state=42)),
        ('gb', GradientBoostingClassifier(random_state=42))
    ], voting='soft')

    # Train and evaluate
    voting_clf.fit(X_train, y_train)
    y_pred_voting = voting_clf.predict(X_test)
    y_pred_voting_proba = voting_clf.predict_proba(X_test)

    voting_results = {
        "Accuracy": accuracy_score(y_test, y_pred_voting),
        "Precision": precision_score(y_test, y_pred_voting, average='macro'),
        "Recall": recall_score(y_test, y_pred_voting, average='macro'),
        "F1-Score": f1_score(y_test, y_pred_voting, average='macro'),
        "AUC-ROC": roc_auc_score(y_test, y_pred_voting_proba, multi_class='ovr')
    }

    #st.write("\nVoting Classifier Results:", voting_results)

    # Save the model
    try:
        with open('voting_clf.pkl', 'wb') as f:
            pickle.dump(voting_clf, f)
    except Exception as e:
        st.error(f"An error occurred while saving the model: {e}")

    return voting_results


# User interface module
def create_user_interface():
    # Load the model
    try:
        with open('voting_clf.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
    except FileNotFoundError:
        st.error("The model file or preprocessing object file was not found. Please train the model first.")
        return

    # Manually define mapping relationships
    gender_mapping = {'F': 0, 'M': 1}

    st.title("Diabetes Early Detection Tool 🩺")
    st.markdown("""
    This tool is designed to assist healthcare professionals in predicting the likelihood of diabetes based on patient data.
    """)

    # User input section
    st.sidebar.header("Patient Data Input")
    gender = st.sidebar.selectbox("Gender", ['F', 'M'])
    age = st.sidebar.number_input("Age", min_value=0, max_value=100, value=30)
    urea = st.sidebar.number_input("Urea (mmol/L)", min_value=0.0, max_value=100.0, value=5.0)
    cr = st.sidebar.number_input("Cr (μmol/L)", min_value=0, max_value=1000, value=50)
    hba1c = st.sidebar.number_input("HbA1c (%)", min_value=0.0, max_value=20.0, value=5.0)
    chol = st.sidebar.number_input("Chol (mmol/L)", min_value=0.0, max_value=10.0, value=5.0)
    tg = st.sidebar.number_input("TG (mmol/L)", min_value=0.0, max_value=10.0, value=1.0)
    hdl = st.sidebar.number_input("HDL (mmol/L)", min_value=0.0, max_value=5.0, value=1.0)
    ldl = st.sidebar.number_input("LDL (mmol/L)", min_value=0.0, max_value=10.0, value=2.0)
    vldl = st.sidebar.number_input("VLDL (mmol/L)", min_value=0.0, max_value=10.0, value=1.0)
    bmi = st.sidebar.number_input("BMI (kg/m²)", min_value=0.0, max_value=60.0, value=25.0)

    # Prediction button
    if st.sidebar.button("Predict"):
        gender_encoded = gender_mapping[gender]
        input_data = np.array([[gender_encoded, age, urea, cr, hba1c, chol, tg, hdl, ldl, vldl, bmi]])
        try:
            # Use the saved preprocessing object for transformation
            input_data = preprocessor.transform(input_data)
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            return

        # Provide real - time feedback on results
        if prediction == 1:
            st.error(f"The patient is at risk of diabetes (Probability: {probability:.2f}).")
        else:
            st.success(f"The patient is not at risk of diabetes (Probability: {probability:.2f}).")

        # Visualize the results
        st.subheader("Prediction Probability")
        fig, ax = plt.subplots()
        ax.bar(["No Diabetes", "Diabetes"], [1 - probability, probability], color=["green", "red"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        st.pyplot(fig)

        # Data analysis visualization
        st.subheader("Data Distribution Analysis")
        data = pd.DataFrame({
            "Feature": ["Gender", "Age", "Urea", "Cr", "HbA1c", "Chol", "TG", "HDL", "LDL", "VLDL", "BMI"],
            "Value": [gender, age, urea, cr, hba1c, chol, tg, hdl, ldl, vldl, bmi]
        })
        data["Feature"] = data["Feature"].astype(str)

        fig, ax = plt.subplots()
        sns.barplot(data=data, x="Feature", y="Value", ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # Customizability: Export function
    def create_download_link(val, filename):
        b64 = base64.b64encode(val).decode()
        return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">Download {filename}</a>'

    if st.sidebar.button("Export Results as PDF"):
        if 'prediction' not in locals() or 'probability' not in locals():
            st.error("Please make a prediction first before exporting the results.")
            return
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Diabetes Prediction Report", ln=True, align="C")
        pdf.cell(200, 10, txt=f"Prediction: {'At Risk' if prediction == 1 else 'Not At Risk'}", ln=True)
        pdf.cell(200, 10, txt=f"Probability: {probability:.2f}", ln=True)

        pdf_output = pdf.output(dest="S").encode("latin-1")
        html = create_download_link(pdf_output, "prediction_report.pdf")
        st.markdown(html, unsafe_allow_html=True)

    # Support and feedback
    st.sidebar.header("Support & Feedback")
    st.sidebar.markdown("""
        For help, please refer to the [User Guide](#) or contact support@example.com.
        """)
    feedback = st.sidebar.text_area("Your Feedback")
    if st.sidebar.button("Submit Feedback"):
        try:
            with open("feedback.txt", "a") as f:
                f.write(feedback + "\n")
            st.sidebar.success("Thank you for your feedback!")
        except Exception as e:
            st.sidebar.error(f"An error occurred while submitting feedback: {e}")
