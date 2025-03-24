# Diabetes Early Detection System

A machine learning-based web application for early diabetes detection using patient health indicators. This system provides accurate predictions and detailed analysis of diabetes risk levels.

![Diabetes Detection](https://img.shields.io/badge/Diabetes-Detection-green)
![Python](https://img.shields.io/badge/Python-3.7+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red)

## Features

- **Multi-Model Ensemble**: Combines Logistic Regression, Random Forest, and Gradient Boosting for robust predictions
- **Three-Class Prediction**: 
  - Non-Diabetic
  - Pre-Diabetic
  - Diabetic
- **Interactive Interface**: User-friendly Streamlit web interface
- **Detailed Reports**: Generates comprehensive PDF reports with prediction results
- **Visual Analytics**: 
  - Probability distribution charts
  - Patient indicators visualization
  - Model performance metrics
- **Feedback System**: Built-in feedback mechanism for system improvement

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Diabetes-early-detection.git
cd Diabetes-early-detection
```


## Usage

1. Run the application:
```bash
streamlit run Front_end.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Enter patient information:
   - Gender
   - Age
   - Urea levels
   - Creatinine levels
   - HbA1c percentage
   - Cholesterol levels (Total, HDL, LDL, VLDL)
   - Triglycerides
   - BMI

4. Click "Start Prediction" to get the results

5. View the prediction results and download the PDF report

## Project Structure

```
Diabetes-early-detection/
├── Front_end.py          # Streamlit web interface
├── back_end.py          # Machine learning models and data processing
├── Dataset of Diabetes .csv  # Training dataset
├── voting_clf.pkl      # Trained ensemble model
├── preprocessor.pkl    # Data preprocessor
└── feedback.txt        # User feedback storage
```

## Model Performance

The system uses an ensemble of three models:
- Logistic Regression
- Random Forest
- Gradient Boosting

Each model's performance is evaluated using multiple metrics:
- Accuracy
- Precision
- Recall
- F1-Score
- AUC-ROC
- Specificity

## Dependencies

- Python 3.7+
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- FPDF

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset source: [Dataset of Diabetes .csv]
- Machine learning models: Scikit-learn
- Web interface: Streamlit
