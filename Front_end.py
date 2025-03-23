import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from back_end import load_and_explore_data, preprocess_data, train_and_evaluate_models, \
    create_and_evaluate_voting_clf, visualize_key_features
from fpdf import FPDF

st.set_option('deprecation.showPyplotGlobalUse', False)

# Set matplotlib font for English
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

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
   .stButton>button,.stDownloadButton>button {
        width: 50%;
        margin: 1rem auto;
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 0.8rem 1.5rem;
        border-radius: 25px;
        font-size: 1.1rem;
        font-weight: bold;
        display: block;
        transition: all 0.3s ease;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
   .stButton>button:hover,.stDownloadButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
   .stButton>button:active,.stDownloadButton>button:active {
        transform: translateY(0);
    }
   .sidebar.sidebar-content {
        background-color: #f0f2f6;
    }
   .button-container {
        display: flex;
        justify-content: center;
        align-items: center;
        width: 100%;
        margin: 1rem 0;
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
    results_df = None
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
                        # New: Call the visualization function and update the results dataframe
                        models = {
                            "Logistic Regression": LogisticRegression(),
                            "Random Forest": RandomForestClassifier(random_state=42),
                            "Gradient Boosting": GradientBoostingClassifier(random_state=42)
                        }
                        results_df = visualize_key_features(models, X_train, y_train, results_df)
                    else:
                        st.error("Model training failed. Please check data or try again.")
                else:
                    st.error("Model training failed. Please check data or try again.")
            else:
                st.error("Data preprocessing failed. Please check data or try again.")
        else:
            st.error("Data loading failed. Please ensure data file exists.")
    else:
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
                # Call the create_and_evaluate_voting_clf function again to get voting_results
                voting_results = create_and_evaluate_voting_clf(X_train, X_test, y_train, y_test)
                if voting_results is not None:
                    st.success("Model training completed!")
                    # Call the visualization function and update the results dataframe
                    models = {
                        "Logistic Regression": LogisticRegression(),
                        "Random Forest": RandomForestClassifier(random_state=42),
                        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
                    }
                    results_df = visualize_key_features(models, X_train, y_train, results_df)
                else:
                    st.error("Model training failed. Please check data or try again.")
            else:
                st.error("Data preprocessing failed. Please check data or try again.")
        else:
            st.error("Data loading failed. Please ensure data file exists.")

    if results_df is not None:
        st.header("📊 Model Performance Evaluation")

        # Display accuracy of all models
        st.subheader("Model Evaluation Metrics")
        st.dataframe(results_df.style.format({
            'Accuracy': '{:.2%}',
            'Precision': '{:.2%}',
            'Recall': '{:.2%}',
            'F1-Score': '{:.2%}',
            'AUC-ROC': '{:.2%}',
            'Key Features': lambda x: x,
            'Feature Importance Scores': lambda x: x,
            'Specificity': '{:.2%}'
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

# Create a container for the prediction button
st.markdown('<div class="button-container">', unsafe_allow_html=True)
if st.button("🔍 Start Prediction", key="predict_button"):
    try:
        # Prepare input data
        gender_mapping = {'F': 0, 'M': 1}
        gender_encoded = gender_mapping[gender]
        input_data = np.array([[gender_encoded, age, urea, cr, hba1c, chol, tg, hdl, ldl, vldl, bmi]])

        # Data preprocessing and prediction
        input_data = preprocessor.transform(input_data)
        prediction = model.predict(input_data)[0]
        probability0 = model.predict_proba(input_data)[0][0]
        probability1 = model.predict_proba(input_data)[0][1]
        probability2 = model.predict_proba(input_data)[0][2]

        # Display prediction results
        st.header("📈 Prediction Results (Ensemble Model)")
        if prediction == 2:
            st.error(f"⚠️ Patient is Diabetic (Probability: {model.predict_proba(input_data)[0][2]:.2%})")
        if prediction == 0:
            st.success(f"✅ Patient is Non - Diabetic (Probability: {model.predict_proba(input_data)[0][0]:.2%})")
        if prediction == 1:
            st.success(f"✅ Patient is Predict - Diabetic (Probability: {model.predict_proba(input_data)[0][1]:.2%})")

        # Create a container for the download button
        st.markdown('<div class="button-container">', unsafe_allow_html=True)
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)

            # Title
            pdf.cell(200, 10, txt="Diabetes Prediction Report", ln=True, align="C")

            # Prediction results
            pdf.cell(200, 10,
                     txt=f"Prediction: {'Diabetic' if prediction == 2 else 'Predict - Diabetic' if prediction == 1 else 'Non - Diabetic'}",
                     ln=True)
            pdf.cell(200, 10,
                     txt=f"Probability: {probability0 if prediction == 0 else probability1 if prediction == 1 else probability2:.2%}",
                     ln=True)

            # Add patient information
            pdf.ln(10)
            pdf.cell(200, 10, txt="Patient Information:", ln=True)
            pdf.cell(200, 10, txt=f"Gender: {gender}", ln=True)
            pdf.cell(200, 10, txt=f"Age: {age}", ln=True)
            pdf.cell(200, 10, txt=f"Urea: {urea} mmol/L", ln=True)
            pdf.cell(200, 10, txt=f"Creatinine: {cr} umol/L", ln=True)
            pdf.cell(200, 10, txt=f"HbA1c: {hba1c}%", ln=True)
            pdf.cell(200, 10, txt=f"Total Cholesterol: {chol} mmol/L", ln=True)
            pdf.cell(200, 10, txt=f"Triglycerides: {tg} mmol/L", ln=True)
            pdf.cell(200, 10, txt=f"HDL Cholesterol: {hdl} mmol/L", ln=True)
            pdf.cell(200, 10, txt=f"LDL Cholesterol: {ldl} mmol/L", ln=True)
            pdf.cell(200, 10, txt=f"VLDL Cholesterol: {vldl} mmol/L", ln=True)
            pdf.cell(200, 10, txt=f"BMI: {bmi} kg/m²", ln=True)

            pdf_output = pdf.output(dest="S").encode("latin-1")

            # Use st.download_button instead of st.button
            st.download_button(
                label="📥 Download Report as PDF",
                data=pdf_output,
                file_name="diabetes_prediction_report.pdf",
                mime="application/pdf"
            )
            st.success("✅ Report generated successfully!")
        except Exception as e:
            st.error(f"Error generating PDF report: {str(e)}")
        st.markdown('</div>', unsafe_allow_html=True)

        # Visualize results
        col1, col2 = st.columns(2)

        with col1:
            fig1, ax1 = plt.subplots()
            ax1.bar(["Non - Diabetic", "Predict - Diabetic", "Diabetic"], [probability0, probability1, probability2],
                    color=["#4CAF50", "#FFEB3B", "#f44336"])
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
            table.scale(1, 3)

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
st.markdown('</div>', unsafe_allow_html=True)

# Feedback Section
st.markdown("---")
st.header("💬 Support & Feedback")
st.markdown("""
    <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 5px; margin-bottom: 1rem;'>
        <h4>Help & Support</h4>
        <p>For help, please refer to the <a href="#">User Guide</a>.</p>
    </div>
    """, unsafe_allow_html=True)

# Create a container for the feedback form
st.markdown("""
    <div style='background-color: #ffffff; padding: 1.5rem; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
    """, unsafe_allow_html=True)

feedback = st.text_area("Your Feedback", height=150, placeholder="Please share your thoughts about the system...")

# Create a container for the submit button
st.markdown('<div class="button-container">', unsafe_allow_html=True)
if st.button("📤 Submit Feedback", key="feedback_button"):
    try:
        with open("feedback.txt", "a", encoding='utf-8') as f:
            f.write(f"{feedback}\n{'=' * 50}\n")  # Add a separator between feedback entries
        st.success("✅ Thank you for your feedback!")
        st.experimental_rerun()  # Clear the form after submission
    except Exception as e:
        st.error(f"An error occurred while submitting feedback: {str(e)}")
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>© 2025 Diabetes Early Detection System | Technical Support</p>
    </div>
    """, unsafe_allow_html=True)