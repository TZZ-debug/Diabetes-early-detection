import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from back_end import load_and_explore_data, preprocess_data, train_and_evaluate_models, \
    create_and_evaluate_voting_clf

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 确保这是第一个Streamlit命令
st.set_page_config(
    page_title="糖尿病早期检测系统",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
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

# 主标题
st.title("🏥 糖尿病早期检测系统")
st.markdown("""
    <div style='background-color: #e6f3ff; padding: 1rem; border-radius: 5px;'>
        <h3>欢迎使用糖尿病早期检测系统</h3>
        <p>本系统基于机器学习模型，通过分析患者的各项指标来预测糖尿病风险。</p>
    </div>
    """, unsafe_allow_html=True)

# 加载并显示模型准确度
try:
    # 检查模型文件是否存在
    if not (os.path.exists('voting_clf.pkl') and os.path.exists('preprocessor.pkl')):
        st.info("模型文件不存在，正在进行模型训练...")
        # 加载数据并训练模型
        data = load_and_explore_data('Dataset of Diabetes .csv')
        if data is not None:
            X_train, X_test, y_train, y_test = preprocess_data(data)
            if X_train is not None:
                # 训练并保存模型
                results_df = train_and_evaluate_models(X_train, X_test, y_train, y_test)
                if results_df is not None:
                    voting_results = create_and_evaluate_voting_clf(X_train, X_test, y_train, y_test)
                    if voting_results is not None:
                        st.success("模型训练完成！")
                    else:
                        st.error("模型训练失败，请检查数据或重试。")
                        st.stop()
                else:
                    st.error("模型训练失败，请检查数据或重试。")
                    st.stop()
            else:
                st.error("数据预处理失败，请检查数据或重试。")
                st.stop()
        else:
            st.error("数据加载失败，请确保数据文件存在。")
            st.stop()

    # 加载训练好的模型
    with open('voting_clf.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)

    # 加载数据并计算模型准确度
    data = load_and_explore_data('Dataset of Diabetes .csv')
    if data is not None:
        X_train, X_test, y_train, y_test = preprocess_data(data)
        if X_train is not None:
            results_df = train_and_evaluate_models(X_train, X_test, y_train, y_test)
            if results_df is not None:
                # 获取集成模型结果
                voting_results = create_and_evaluate_voting_clf(X_train, X_test, y_train, y_test)
                if voting_results is not None:
                    # 将集成模型结果添加到results_df
                    results_df.loc['Ensemble Model'] = voting_results

                st.header("📊 模型性能评估")

                # 显示所有模型的准确度
                st.subheader("各模型评估指标")
                st.dataframe(results_df.style.format({
                    'Accuracy': '{:.2%}',
                    'Precision': '{:.2%}',
                    'Recall': '{:.2%}',
                    'F1-Score': '{:.2%}',
                    'AUC-ROC': '{:.2%}'
                }))

                # 单独展示集成模型的准确度
                st.subheader("集成模型性能")
                ensemble_metrics = pd.DataFrame({
                    '准确率': [voting_results['Accuracy']],
                    '精确率': [voting_results['Precision']],
                    '召回率': [voting_results['Recall']],
                    'F1分数': [voting_results['F1-Score']],
                    'AUC-ROC': [voting_results['AUC-ROC']]
                }, index=['集成模型'])

                st.dataframe(ensemble_metrics.style.format('{:.2%}'))

                # 可视化模型性能
                fig, ax = plt.subplots(figsize=(10, 6))
                results_df['Accuracy'].plot(kind='bar', ax=ax)
                plt.title('各模型准确率对比', fontsize=12)
                plt.xlabel('模型', fontsize=10)
                plt.ylabel('准确率', fontsize=10)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
except Exception as e:
    st.error(f"模型加载或训练过程中发生错误: {str(e)}")
    st.stop()

# 主界面
st.header("👤 患者信息输入")
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("性别", ['F', 'M'])
    age = st.number_input("年龄", min_value=0, max_value=100, value=30)
    urea = st.number_input("尿素 (mmol/L)", min_value=0.0, max_value=100.0, value=5.0)
    cr = st.number_input("肌酐 (μmol/L)", min_value=0, max_value=1000, value=50)
    hba1c = st.number_input("糖化血红蛋白 (%)", min_value=0.0, max_value=20.0, value=5.0)
    chol = st.number_input("总胆固醇 (mmol/L)", min_value=0.0, max_value=10.0, value=5.0)

with col2:
    tg = st.number_input("甘油三酯 (mmol/L)", min_value=0.0, max_value=10.0, value=1.0)
    hdl = st.number_input("高密度脂蛋白胆固醇 (mmol/L)", min_value=0.0, max_value=5.0, value=1.0)
    ldl = st.number_input("低密度脂蛋白胆固醇 (mmol/L)", min_value=0.0, max_value=10.0, value=2.0)
    vldl = st.number_input("极低密度脂蛋白胆固醇 (mmol/L)", min_value=0.0, max_value=10.0, value=1.0)
    bmi = st.number_input("身体质量指数 (kg/m²)", min_value=0.0, max_value=60.0, value=25.0)

# 预测按钮
st.markdown("""
    <div style='background-color: #f0f2f6; padding: 1rem; border-radius: 5px; margin-bottom: 1rem;'>
        <h4>⚠️ 注意：当前使用集成模型进行预测</h4>
        <p>集成模型综合了逻辑回归、随机森林和梯度提升三个模型的优点，通常能提供更稳定和准确的预测结果。</p>
    </div>
    """, unsafe_allow_html=True)

if st.button("开始预测", key="predict_button"):
    try:
        # 准备输入数据
        gender_mapping = {'F': 0, 'M': 1}
        gender_encoded = gender_mapping[gender]
        input_data = np.array([[gender_encoded, age, urea, cr, hba1c, chol, tg, hdl, ldl, vldl, bmi]])

        # 数据预处理和预测
        input_data = preprocessor.transform(input_data)
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        # 显示预测结果
        st.header("📈 预测结果 (集成模型)")
        if prediction == 1:
            st.error(f"⚠️ 该患者存在糖尿病风险 (概率: {probability:.2%})")
        else:
            st.success(f"✅ 该患者目前没有糖尿病风险 (概率: {probability:.2%})")

        # 可视化结果
        col1, col2 = st.columns(2)

        with col1:
            fig1, ax1 = plt.subplots()
            ax1.bar(["无风险", "有风险"], [1 - probability, probability],
                    color=["#4CAF50", "#f44336"])
            ax1.set_ylim(0, 1)
            ax1.set_ylabel("概率", fontsize=10)
            ax1.set_title("糖尿病风险预测概率", fontsize=12)
            st.pyplot(fig1)

        with col2:
            # 创建患者数据可视化
            data = pd.DataFrame({
                "指标": ["性别", "年龄", "尿素", "肌酐", "糖化血红蛋白", "总胆固醇",
                         "甘油三酯", "HDL", "LDL", "VLDL", "BMI"],
                "数值": [gender, str(age), str(urea), str(cr), str(hba1c), str(chol),
                         str(tg), str(hdl), str(ldl), str(vldl), str(bmi)]
            })

            fig2, ax2 = plt.subplots(figsize=(10, 6))
            sns.barplot(data=data, x="指标", y="数值", ax=ax2)
            plt.xticks(rotation=45, fontsize=10)
            plt.yticks(fontsize=10)
            ax2.set_title("患者指标分布", fontsize=12)
            ax2.set_xlabel("数值", fontsize=10)
            ax2.set_ylabel("指标", fontsize=10)
            plt.tight_layout()
            st.pyplot(fig2)

    except Exception as e:
        st.error(f"预测过程中发生错误: {str(e)}")

# 页脚
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>© 2024 糖尿病早期检测系统 | 技术支持</p>
    </div>
    """, unsafe_allow_html=True)