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


# 数据加载与探索性数据分析 (EDA) 模块
def load_and_explore_data(file_path):
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"数据集文件 '{file_path}' 未找到，请检查文件名和路径。")
        return None

    # 手动定义映射关系
    gender_mapping = {'F': 0, 'M': 1}
    class_mapping = {'N': 0, 'P': 1, 'Y': 2}

    # 对Gender和Class列进行编码
    data['Gender'] = data['Gender'].str.strip().map(gender_mapping)
    data['CLASS'] = data['CLASS'].str.strip().map(class_mapping)

    # 查看数据的基本信息
    #st.write("\nData Description:")
    #st.write(data.describe())

    # 检查缺失值
    #st.write("\nMissing Values:")
    #st.write(data.isnull().sum())

    # 数据分布可视化
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

    # 相关性矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    st.pyplot()
    '''

    return data


# 数据预处理模块
def preprocess_data(data):
    if data is None:
        return None, None

    # 特征与标签分离，假设CLASS是目标变量
    X = data.drop(columns=['CLASS']).drop(columns=['ID']).drop(columns=['No_Pation'])
    y = data['CLASS']

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 保存预处理对象
    with open('preprocessor.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # 数据集划分
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test


# 模型开发与评估模块
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    if X_train is None or X_test is None or y_train is None or y_test is None:
        return None, None

    # 定义模型
    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }

    # 训练与评估
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

    # 结果展示
    results_df = pd.DataFrame(results).T
    #st.write("\nModel Evaluation Results:")
    #st.write(results_df)

    return results_df


# 集成学习模块
def create_and_evaluate_voting_clf(X_train, X_test, y_train, y_test):
    if X_train is None or X_test is None or y_train is None or y_test is None:
        return None

    # 创建集成模型
    voting_clf = VotingClassifier(estimators=[
        ('lr', LogisticRegression()),
        ('rf', RandomForestClassifier(random_state=42)),
        ('gb', GradientBoostingClassifier(random_state=42))
    ], voting='soft')

    # 训练与评估
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

    # 保存模型
    try:
        with open('voting_clf.pkl', 'wb') as f:
            pickle.dump(voting_clf, f)
    except Exception as e:
        st.error(f"保存模型时发生错误: {e}")

    return voting_results


# 用户界面模块
def create_user_interface():
    # 加载模型
    try:
        with open('voting_clf.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
    except FileNotFoundError:
        st.error("模型文件或预处理对象文件未找到，请先训练模型。")
        return

    # 手动定义映射关系
    gender_mapping = {'F': 0, 'M': 1}

    st.title("Diabetes Early Detection Tool 🩺")
    st.markdown("""
    This tool is designed to assist healthcare professionals in predicting the likelihood of diabetes based on patient data.
    """)

    # 用户输入部分
    st.sidebar.header("Patient Data Input")
    gender = st.sidebar.selectbox("Gender（性别）", ['F', 'M'])
    age = st.sidebar.number_input("Age（年龄）", min_value=0, max_value=100, value=30)
    urea = st.sidebar.number_input("Urea（尿素，mmol/L）", min_value=0.0, max_value=100.0, value=5.0)
    cr = st.sidebar.number_input("Cr（肌酐，μmol/L）", min_value=0, max_value=1000, value=50)
    hba1c = st.sidebar.number_input("HbA1c（糖化血红蛋白，%）", min_value=0.0, max_value=20.0, value=5.0)
    chol = st.sidebar.number_input("Chol（总胆固醇，mmol/L）", min_value=0.0, max_value=10.0, value=5.0)
    tg = st.sidebar.number_input("TG（甘油三酯，mmol/L）", min_value=0.0, max_value=10.0, value=1.0)
    hdl = st.sidebar.number_input("HDL（高密度脂蛋白胆固醇，mmol/L）", min_value=0.0, max_value=5.0, value=1.0)
    ldl = st.sidebar.number_input("LDL（低密度脂蛋白胆固醇，mmol/L）", min_value=0.0, max_value=10.0, value=2.0)
    vldl = st.sidebar.number_input("VLDL（极低密度脂蛋白胆固醇，mmol/L）", min_value=0.0, max_value=10.0, value=1.0)
    bmi = st.sidebar.number_input("BMI（身体质量指数，kg/m²）", min_value=0.0, max_value=60.0, value=25.0)

    # 预测按钮
    if st.sidebar.button("Predict"):
        gender_encoded = gender_mapping[gender]
        input_data = np.array([[gender_encoded, age, urea, cr, hba1c, chol, tg, hdl, ldl, vldl, bmi]])
        try:
            # 使用保存的预处理对象进行转换
            input_data = preprocessor.transform(input_data)
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]
        except Exception as e:
            st.error(f"预测时发生错误: {e}")
            return

        # 实时反馈结果
        if prediction == 1:
            st.error(f"The patient is at risk of diabetes (Probability: {probability:.2f}).")
        else:
            st.success(f"The patient is not at risk of diabetes (Probability: {probability:.2f}).")

        # 可视化结果
        st.subheader("Prediction Probability")
        fig, ax = plt.subplots()
        ax.bar(["No Diabetes", "Diabetes"], [1 - probability, probability], color=["green", "red"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        st.pyplot(fig)

        # 数据分析可视化
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

    # 可定制性：导出功能
    def create_download_link(val, filename):
        b64 = base64.b64encode(val).decode()
        return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">Download {filename}</a>'

    if st.sidebar.button("Export Results as PDF"):
        if 'prediction' not in locals() or 'probability' not in locals():
            st.error("请先进行预测，再导出结果。")
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

    # 支持与反馈
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
            st.sidebar.error(f"提交反馈时发生错误: {e}")
