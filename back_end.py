import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import streamlit as st
import pickle

# Module for data loading and exploratory data analysis (EDA)
def load_and_explore_data(file_path):
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"The dataset file '{file_path}' was not found. Please check the file name and path.")
        return None

    # Manually define mapping relationships
    gender_mapping = {'F': 0, 'M': 1}
    class_mapping = {'N': 0, 'P': 1, 'Y': 2}

    # Encode the 'Gender' and 'CLASS' columns
    data['Gender'] = data['Gender'].str.strip().map(gender_mapping)
    data['CLASS'] = data['CLASS'].str.strip().map(class_mapping)

    return data

# Module for data preprocessing
def preprocess_data(data):
    if data is None:
        return None, None

    # Separate features and labels. Assume 'CLASS' is the target variable
    X = data.drop(columns=['CLASS', 'ID', 'No_Pation'])
    y = data['CLASS']

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Convert X_scaled to a DataFrame and retain feature names
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # Save the preprocessing object
    with open('preprocessor.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test

# Module for model development and evaluation
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    if X_train is None or X_test is None or y_train is None or y_test is None:
        return None

    # Define models
    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "The ensemble model": VotingClassifier(estimators=[
            ('lr', LogisticRegression()),
            ('rf', RandomForestClassifier(random_state=42)),
            ('gb', GradientBoostingClassifier(random_state=42))
        ], voting='soft')
    }

    # Train and evaluate models
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        # Calculate specificity for each class and take the average
        cm = confusion_matrix(y_test, y_pred)
        num_classes = cm.shape[0]
        specificities = []
        for i in range(num_classes):
            tn = cm.sum() - cm[i].sum() - cm[:, i].sum() + cm[i, i]
            fp = cm[:, i].sum() - cm[i, i]
            specificity = tn / (tn + fp)
            specificities.append(specificity)
        avg_specificity = sum(specificities) / num_classes

        results[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average='macro'),
            "Recall": recall_score(y_test, y_pred, average='macro'),
            "F1-Score": f1_score(y_test, y_pred, average='macro'),
            "AUC-ROC": roc_auc_score(y_test, y_pred_proba, multi_class='ovr'),
            "Specificity": avg_specificity
        }

        if name == "The ensemble model":
            # For the ensemble model, don't recalculate feature importance (since it's a combination of multiple models)
            results[name]["Key Features"] = ""
            results[name]["Feature Importance Scores"] = ""
        else:
            # Extract feature names
            feature_names = X_train.columns
            if name == "Logistic Regression":
                # First, train the logistic regression model
                model.fit(X_train, y_train)
                # Use the absolute values of coefficients as an approximate measure of feature importance
                feature_importance = pd.Series(
                    abs(model.coef_[0]), index=feature_names).sort_values(ascending=False)
            elif name in ["Random Forest", "Gradient Boosting"]:
                # First, train the model
                model.fit(X_train, y_train)
                # Get the feature importance scores of the model
                feature_importance = pd.Series(
                    model.feature_importances_, index=feature_names).sort_values(ascending=False)

            results[name]["Key Features"] = ', '.join(feature_importance.index[:5])
            results[name]["Feature Importance Scores"] = ', '.join(
                str(round(score, 4)) for score in feature_importance.values[:5])

    # Display results
    results_df = pd.DataFrame(results).T
    return results_df

# Ensemble learning module. You can keep the original function or adjust it as needed
def create_and_evaluate_voting_clf(X_train, X_test, y_train, y_test):
    if X_train is None or X_test is None or y_train is None or y_test is None:
        return None

    # Create an ensemble model
    voting_clf = VotingClassifier(estimators=[
        ('lr', LogisticRegression()),
        ('rf', RandomForestClassifier(random_state=42)),
        ('gb', GradientBoostingClassifier(random_state=42))
    ], voting='soft')

    # Train and evaluate the ensemble model
    voting_clf.fit(X_train, y_train)
    y_pred_voting = voting_clf.predict(X_test)
    y_pred_voting_proba = voting_clf.predict_proba(X_test)

    # Calculate specificity
    cm = confusion_matrix(y_test, y_pred_voting)
    num_classes = cm.shape[0]
    specificities = []
    for i in range(num_classes):
        tn = cm.sum() - cm[i].sum() - cm[:, i].sum() + cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        specificity = tn / (tn + fp)
        specificities.append(specificity)
    avg_specificity = sum(specificities) / num_classes

    voting_results = {
        "Accuracy": accuracy_score(y_test, y_pred_voting),
        "Precision": precision_score(y_test, y_pred_voting, average='macro'),
        "Recall": recall_score(y_test, y_pred_voting, average='macro'),
        "F1-Score": f1_score(y_test, y_pred_voting, average='macro'),
        "AUC-ROC": roc_auc_score(y_test, y_pred_voting_proba, multi_class='ovr'),
        "Specificity": avg_specificity
    }

    # Save the model
    try:
        with open('voting_clf.pkl', 'wb') as f:
            pickle.dump(voting_clf, f)
    except Exception as e:
        st.error(f"An error occurred while saving the model: {e}")

    return voting_results

# New function to visualize key features
def visualize_key_features(models, X_train, y_train, results_df):
    # Extract feature names
    feature_names = X_train.columns

    # Iterate through models
    for name, model in models.items():
        if name == "Logistic Regression":
            # First, train the logistic regression model
            model.fit(X_train, y_train)
            # Use the absolute values of coefficients as an approximate measure of feature importance
            feature_importance = pd.Series(
                abs(model.coef_[0]), index=feature_names).sort_values(ascending=False)
        elif name in ["Random Forest", "Gradient Boosting"]:
            # First, train the model
            model.fit(X_train, y_train)
            # Get the feature importance scores of the model
            feature_importance = pd.Series(
                model.feature_importances_, index=feature_names).sort_values(ascending=False)
        else:
            continue

        # Plot a bar chart of feature importance and reduce the size of the graph
        plt.figure(figsize=(6, 4))  # Set the width to 6 inches and height to 4 inches here, which can be adjusted as needed
        sns.barplot(x=feature_importance.values, y=feature_importance.index)

        # Set the font size of axis labels
        plt.xlabel('Feature Importance Score', fontsize=8)
        plt.ylabel('Features', fontsize=8)

        # Set the font size of the title
        plt.title(f'Feature Importance for {name}', fontsize=10)

        # Set the font size of tick labels
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)

        # Display the graph using st.pyplot()
        st.pyplot()

        # Add feature importance to the results DataFrame
        results_df.loc[name, 'Key Features'] = ', '.join(feature_importance.index[:5])
        results_df.loc[name, 'Feature Importance Scores'] = ', '.join(
            str(round(score, 4)) for score in feature_importance.values[:5])

    return results_df