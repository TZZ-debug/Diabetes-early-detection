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

    # 将 X_scaled 转换为 DataFrame，保留特征名称
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # Save the preprocessing object
    with open('preprocessor.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test

# Model development and evaluation module
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

    # Train and evaluate
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        # 计算每个类别的特异性并取平均值
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
            # 对于集成模型，不重复计算特征重要性（因为它是多个模型的组合）
            results[name]["Key Features"] = ""
            results[name]["Feature Importance Scores"] = ""
        else:
            # 提取特征名称
            feature_names = X_train.columns
            if name == "Logistic Regression":
                # 先训练逻辑回归模型
                model.fit(X_train, y_train)
                # 使用系数的绝对值作为特征重要性的近似度量
                feature_importance = pd.Series(
                    abs(model.coef_[0]), index=feature_names).sort_values(ascending=False)
            elif name in ["Random Forest", "Gradient Boosting"]:
                # 先训练模型
                model.fit(X_train, y_train)
                # 获取模型的特征重要性得分
                feature_importance = pd.Series(
                    model.feature_importances_, index=feature_names).sort_values(ascending=False)

            results[name]["Key Features"] = ', '.join(feature_importance.index[:5])
            results[name]["Feature Importance Scores"] = ', '.join(
                str(round(score, 4)) for score in feature_importance.values[:5])

    # Display results
    results_df = pd.DataFrame(results).T
    return results_df

# Ensemble learning module，这里可以保留原函数，也可以根据需要调整
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

    # 计算特异性
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

# 新增可视化关键特征的函数
def visualize_key_features(models, X_train, y_train, results_df):
    # 提取特征名称
    feature_names = X_train.columns

    # 遍历模型
    for name, model in models.items():
        if name == "Logistic Regression":
            # 先训练逻辑回归模型
            model.fit(X_train, y_train)
            # 使用系数的绝对值作为特征重要性的近似度量
            feature_importance = pd.Series(
                abs(model.coef_[0]), index=feature_names).sort_values(ascending=False)
        elif name in ["Random Forest", "Gradient Boosting"]:
            # 先训练模型
            model.fit(X_train, y_train)
            # 获取模型的特征重要性得分
            feature_importance = pd.Series(
                model.feature_importances_, index=feature_names).sort_values(ascending=False)
        else:
            continue

        # 绘制特征重要性柱状图，调小图形尺寸
        plt.figure(figsize=(6, 4))  # 这里将宽度设为 6 英寸，高度设为 4 英寸，可按需调整
        sns.barplot(x=feature_importance.values, y=feature_importance.index)

        # 设置坐标轴标签字体大小
        plt.xlabel('Feature Importance Score', fontsize=8)
        plt.ylabel('Features', fontsize=8)

        # 设置标题字体大小
        plt.title(f'Feature Importance for {name}', fontsize=10)

        # 设置刻度标签字体大小
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)

        # 使用 st.pyplot() 显示图形
        st.pyplot()

        # 将特征重要性添加到结果数据框中
        results_df.loc[name, 'Key Features'] = ', '.join(feature_importance.index[:5])
        results_df.loc[name, 'Feature Importance Scores'] = ', '.join(
            str(round(score, 4)) for score in feature_importance.values[:5])

    return results_df
