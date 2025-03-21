import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from back_end import load_and_explore_data, preprocess_data, train_and_evaluate_models, \
    create_and_evaluate_voting_clf

# Set matplotlib font for English
plt.rcParams['font.sans-serif'] = ['Arial']  # For English labels
plt.rcParams['axes.unicode_minus'] = False  # For negative signs

# Ensure this is the first Streamlit command
st.set_page_config(
    page_title="Diabetes Early Detection System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styles
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    </style>
    """, unsafe_allow_html=True)

# Main title
st.title("🏥 Diabetes Early Detection System")
st.markdown("""
    <div style='background-color: #e6f3ff; padding: 1rem; border-radius: 5px;'>
        <h3>Welcome to Diabetes Early Detection System</h3>
        <p>This system uses machine learning models to predict diabetes risk by analyzing various patient indicators.</p>
    </div>
    """, unsafe_allow_html=True)

# Load and display model accuracy
try:
    # Check if model files exist
    if not (os.path.exists('voting_clf.pkl') and os.path.exists('preprocessor.pkl')):
        st.info("Model files not found, starting model training...")
        # Load data and train model
        data = load_and_explore_data('Dataset of Diabetes .csv')
        if data is not None:
            X_train, X_test, y_train, y_test = preprocess_data(data)
            if X_train is not None:
                # Train and save model
                results_df = train_and_evaluate_models(X_train, X_test, y_train, y_test)
                if results_df is not None:
                    voting_results = create_and_evaluate_voting_clf(X_train, X_test, y_train, y_test)
                    if voting_results is not None:
                        st.success("Model training completed!")
                    else:
                        st.error("Model training failed. Please check data or try again.")
                        st.stop()
                else:
                    st.error("Model training failed. Please check data or try again.")
                    st.stop()
            else:
                st.error("Data preprocessing failed. Please check data or try again.")
                st.stop()
        else:
            st.error("Data loading failed. Please ensure data file exists.")
            st.stop()

    # Load trained model
    with open('voting_clf.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)

    # Load data and calculate model accuracy
    data = load_and_explore_data('Dataset of Diabetes .csv')
    if data is not None:
        X_train, X_test, y_train, y_test = preprocess_data(data)
        if X_train is not None:
            results_df = train_and_evaluate_models(X_train, X_test, y_train, y_test)
            if results_df is not None:
                # Get ensemble model results
                voting_results = create_and_evaluate_voting_clf(X_train, X_test, y_train, y_test)
                if voting_results is not None:
                    # Add ensemble model results to results_df
                    results_df.loc['Ensemble Model'] = voting_results

                st.header("📊 Model Performance Evaluation")

                # Display accuracy of all models
                st.subheader("Model Evaluation Metrics")
                st.dataframe(results_df.style.format({
                    'Accuracy': '{:.2%}',
                    'Precision': '{:.2%}',
                    'Recall': '{:.2%}',
                    'F1-Score': '{:.2%}',
                    'AUC-ROC': '{:.2%}'
                }))

                # Visualize model performance
                fig, ax = plt.subplots(figsize=(10, 6))
                results_df['Accuracy'].plot(kind='bar', ax=ax)
                plt.title('Model Accuracy Comparison', fontsize=12)
                plt.xlabel('Model', fontsize=10)
                plt.ylabel('Accuracy', fontsize=10)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
except Exception as e:
    st.error(f"Error in model loading or training process: {str(e)}")
    st.stop()

# Main interface
st.header("👤 Patient Information Input")
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ['F', 'M'])
    age = st.number_input("Age", min_value=0, max_value=100, value=30)
    urea = st.number_input("Urea (mmol/L)", min_value=0.0, max_value=100.0, value=5.0)
    cr = st.number_input("Creatinine (μmol/L)", min_value=0, max_value=1000, value=50)
    hba1c = st.number_input("HbA1c (%)", min_value=0.0, max_value=20.0, value=5.0)
    chol = st.number_input("Total Cholesterol (mmol/L)", min_value=0.0, max_value=10.0, value=5.0)

with col2:
    tg = st.number_input("Triglycerides (mmol/L)", min_value=0.0, max_value=10.0, value=1.0)
    hdl = st.number_input("HDL Cholesterol (mmol/L)", min_value=0.0, max_value=5.0, value=1.0)
    ldl = st.number_input("LDL Cholesterol (mmol/L)", min_value=0.0, max_value=10.0, value=2.0)
    vldl = st.number_input("VLDL Cholesterol (mmol/L)", min_value=0.0, max_value=10.0, value=1.0)
    bmi = st.number_input("Body Mass Index (kg/m²)", min_value=0.0, max_value=60.0, value=25.0)

# Prediction button
st.markdown("""
    <div style='background-color: #f0f2f6; padding: 1rem; border-radius: 5px; margin-bottom: 1rem;'>
        <h4>⚠️ Note: Currently using ensemble model for prediction</h4>
        <p>The ensemble model combines the advantages of logistic regression, random forest, and gradient boosting models to provide more stable and accurate predictions.</p>
    </div>
    """, unsafe_allow_html=True)

if st.button("Start Prediction", key="predict_button"):
    try:
        # Prepare input data
        gender_mapping = {'F': 0, 'M': 1}
        gender_encoded = gender_mapping[gender]
        input_data = np.array([[gender_encoded, age, urea, cr, hba1c, chol, tg, hdl, ldl, vldl, bmi]])

        # Data preprocessing and prediction
        input_data = preprocessor.transform(input_data)
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        # Display prediction results
        st.header("📈 Prediction Results (Ensemble Model)")
        if prediction == 1:
            st.error(f"⚠️ Patient is at risk of diabetes (Probability: {probability:.2%})")
        else:
            st.success(f"✅ Patient is not at risk of diabetes (Probability: {probability:.2%})")

        # Visualize results
        col1, col2 = st.columns(2)

        with col1:
            fig1, ax1 = plt.subplots()
            ax1.bar(["No Risk", "At Risk"], [1 - probability, probability],
                    color=["#4CAF50", "#f44336"])
            ax1.set_ylim(0, 1)
            ax1.set_ylabel("Probability", fontsize=10)
            ax1.set_title("Diabetes Risk Prediction Probability", fontsize=12)
            st.pyplot(fig1)

        with col2:
            # Create patient data visualization
            data = pd.DataFrame({
                "Indicator": ["Gender", "Age", "Urea", "Creatinine", "HbA1c", "Total Cholesterol",
                             "Triglycerides", "HDL", "LDL", "VLDL", "BMI"],
                "Value": [gender, str(age), str(urea), str(cr), str(hba1c), str(chol),
                         str(tg), str(hdl), str(ldl), str(vldl), str(bmi)]
            })

            fig2, ax2 = plt.subplots(figsize=(10, 8))
            ax2.axis('off')

            # Create a table
            table = ax2.table(cellText=data.values, colLabels=data.columns, cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1, 3)  # Adjust the table size

            # Set the background color and bold font for the table header
            for (row, col), cell in table.get_celld().items():
                if row == 0:
                    cell.set_facecolor('#40466e')
                    cell.get_text().set_fontweight('bold')
                    cell.get_text().set_color('white')

            # Set the background color and bold font for the table content
            for (row, col), cell in table.get_celld().items():
                if row > 0:
                    if row % 2 == 0:
                        cell.set_facecolor('#f2f2f2')
                    else:
                        cell.set_facecolor('#ffffff')
                    cell.get_text().set_fontweight('bold')

            # Adjust the border style of the cells
            for key, cell in table.get_celld().items():
                cell.set_edgecolor('lightgray')

            plt.title("Patient Indicators Distribution", fontsize=15)
            plt.tight_layout()
            st.pyplot(fig2)

    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>© 2024 Diabetes Early Detection System | Technical Support</p>
    </div>
    """, unsafe_allow_html=True)