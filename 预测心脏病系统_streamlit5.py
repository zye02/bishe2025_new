import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 页面配置
st.set_page_config(page_title="心脏病预测", layout="wide")

# 样式设计
sns.set_style("whitegrid")


def set_background_image(image_path='background.jpg'):
    """
    设置页面背景图为本地图片，并调整其大小和位置
    :param image_path: 图片路径（相对于脚本文件）
    """
    import base64

    with open(image_path, "rb") as f:
        encoded_str = base64.b64encode(f.read()).decode()

    # 添加全局样式，包括背景图片的设置、输入框背景颜色和所有文字的加粗
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/jpg;base64,{encoded_str});
            background-size: contain; /* 调整背景图片大小 */
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: 70% center;
        }}

        /* 设置所有输入框的背景颜色为白色 */
        input, textarea {{
            background-color: white !important;
            font-weight: normal !important; /* 保证输入框内的文字不是加粗 */
        }}

        /* 加粗所有其他地方的文字 */
        body, h1, h2, h3, h4, h5, h6, p, span, label, button {{
            font-weight: bold !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# 确保 users.json 文件存在
def ensure_users_file_exists():
    if not os.path.exists('users.json'):
        with open('users.json', 'w') as f:
            json.dump({}, f)


# 登录验证函数（检查用户名和密码是否匹配）
def login_user(username, password):
    ensure_users_file_exists()
    with open('users.json', 'r') as f:
        users = json.load(f)
    return users.get(username, {}).get("password") == password

# 注册新用户函数（包括性别和年龄）
def register_user(username, password, gender, age):
    ensure_users_file_exists()
    with open('users.json', 'r') as f:
        users = json.load(f)

    if username in users:
        return False  # 用户名已存在

    users[username] = {
        "password": password,
        "gender": gender,
        "age": age
    }

    with open('users.json', 'w') as f:
        json.dump(users, f)

    return True

# 加载数据
@st.cache_data
def load_and_clean_data():
    df = pd.read_excel('heart_0513.xlsx')
    rows_with_nan = df[df.isnull().any(axis=1)]
    df_cleaned = df.dropna()

    def remove_outliers(df):
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        df_no_outliers = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
        return df_no_outliers

    df_final = remove_outliers(df_cleaned)
    return df_final

# 训练模型
@st.cache_resource
def train_model(df):
    X = df[['age', 'sex', 'trestbps', 'chol', 'fbs', 'thalach', 'exang', 'thal']]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
        'n_estimators': [100, 50],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    best_rf = grid_search.best_estimator_

    # 预测并评估模型
    y_pred = best_rf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    return best_rf, report

# =============================
def render_login_register():
    st.title("🔐 登录 / 注册")
    option = st.selectbox("请选择操作", ["登录", "注册"])

    if option == "登录":
        st.subheader("请登录")
        username = st.text_input("用户名")
        password = st.text_input("密码", type="password")
        if st.button("登录"):
            if login_user(username, password):
                st.session_state['logged_in'] = True
                st.session_state['current_user'] = username
                st.success("登录成功！")
            else:
                st.error("用户名或密码错误")

    elif option == "注册":
        st.subheader("创建新账户")
        new_username = st.text_input("新用户名")
        new_password = st.text_input("新密码", type="password")
        confirm_password = st.text_input("确认密码", type="password")
        gender = st.selectbox("性别", ["男", "女"])
        age = st.number_input("年龄", min_value=0, max_value=120, value=18)

        if st.button("注册"):
            if new_password != confirm_password:
                st.error("两次输入的密码不一致！")
            elif len(new_password) < 6:
                st.warning("密码至少需要6位字符！")
            else:
                if register_user(new_username, new_password, gender, age):
                    st.success("注册成功！请返回登录页登录。")
                else:
                    st.warning("用户名已存在，请换一个。")

# 个人资料页面
# =============================
def render_profile():
    st.title("🧾 个人资料")

    username = st.session_state.get("current_user")
    if not username:
        st.warning("请先登录查看个人资料。")
        return

    with open('users.json', 'r') as f:
        users = json.load(f)

    user_info = users.get(username)
    if not user_info:
        st.error("用户信息不存在，请重新登录。")
        return

    st.markdown(f"**用户名：** {username}")
    st.markdown(f"**性别：** {user_info['gender']}")
    st.markdown(f"**年龄：** {user_info['age']}")

    st.markdown("---")
    st.subheader("🔐 更改密码")

    old_password = st.text_input("当前密码", type="password")
    new_password = st.text_input("新密码", type="password")
    confirm_password = st.text_input("确认新密码", type="password")

    if st.button("更改密码"):
        if old_password != user_info["password"]:
            st.error("当前密码错误！")
        elif new_password != confirm_password:
            st.error("两次输入的新密码不一致！")
        elif new_password.strip() == "":
            st.error("新密码不能为空！")
        else:
            users[username]["password"] = new_password
            with open('users.json', 'w') as f:
                json.dump(users, f)
            st.success("密码修改成功！")

import streamlit as st
import streamlit as st

def sidebar_navigation():
    # 自定义CSS样式
    st.markdown("""
    <style>
        .nav-title {
            background-color: #d6eaff; /* 设置为白色或任何你希望的颜色 */
            padding: 10px;
            border: 2px solid #d6eaff; /* 使用边框来框起文字 */
            margin-bottom: 20px;
            text-align: left; /* 左对齐 */
            font-size: 18px;
            font-weight: bold;
            color: #333333;
            width: 100%; /* 确保宽度占满容器 */
            box-sizing: border-box; /* 包括padding和border在内的宽度计算 */
        }
        .nav-button {
            background-color: #d6eaff; /* 浅蓝色 */
            padding: 12px;
            border-radius: 8px;
            margin: 8px 0;
            text-align: left; /* 改成left让图标和文本左对齐 */
            font-weight: bold;
            color: #003366;
            cursor: pointer;
            transition: background-color 0.3s ease;
            width: 100%; /* 确保按钮宽度占满容器 */
            box-sizing: border-box; /* 包括padding和border在内的宽度计算 */
        }
        .nav-button:hover {
            background-color: #a3d0ff;
        }
    </style>
    """, unsafe_allow_html=True)

    # 渲染导航标题
    st.sidebar.markdown('<div class="nav-title">导航</div>', unsafe_allow_html=True)

    # 页面选项和图标
    pages = {
        "登录与注册": "🔐 登录 / 注册",
        "数据分析与可视化": "📊 数据分析与可视化",
        "心脏病预测": "🫀 心脏病概率预测",
        "个人信息": "🧾 个人资料"
    }

    # 渲染每个页面按钮
    for page_key, label in pages.items():
        if st.sidebar.button(label, key=f"nav_{page_key}", use_container_width=True):
            st.session_state["page"] = page_key

    # 默认页面
    if "page" not in st.session_state:
        st.session_state["page"] = "登录与注册"

    return st.session_state.get("page", "登录与注册")


import matplotlib.pyplot as plt
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def render_visualizations(df):
    st.title("📊 数据分析与可视化")

    # =====================
    # 字段映射：英文列名 → 简洁中文名（带详细描述）
    # =====================
    var_name_map = {
        'age': '年龄（岁）',
        'sex': '性别（0=女，1=男）',
        'trestbps': '静息血压（mm Hg）',
        'chol': '血清胆固醇（mg/dl）',
        'fbs': '空腹血糖 > 120 mg/dl（0=否，1=是）',
        'thalach': '最大心率',
        'exang': '运动诱发心绞痛（0=否，1=是）',
        'thal': '地中海贫血类型（0=正常，1=固定缺陷，2=可逆缺陷）',
        'target': '是否患病'
    }

    continuous_vars = ['age', 'trestbps', 'chol', 'thalach']
    categorical_vars = ['sex', 'fbs', 'exang', 'thal']  # 只保留指定的分类变量

    # 自动添加中文名称
    def get_chinese_name(var):
        return var_name_map.get(var, var)

    # =====================
    # a) 连续型变量直方图
    # =====================
    st.subheader("a) 连续型变量分布直方图")
    cols = st.columns(2)
    for i, var in enumerate(continuous_vars):
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.histplot(df[var], kde=True, ax=ax, color='skyblue')
        ax.set_title(var)
        cols[i % 2].pyplot(fig)
        cols[i % 2].caption(f"`{var}` → {get_chinese_name(var)}")  # 添加中文注释

    # =====================
    # b) 分类变量饼图（只保留 sex, fbs, exang, thal）
    # =====================
    st.subheader("b) 分类变量分布饼图")
    cols = st.columns(2)
    pie_colors = ['skyblue', 'lightgreen', 'salmon', 'gold', 'violet', 'orange', 'cyan']

    for i, var in enumerate(categorical_vars):
        fig, ax = plt.subplots(figsize=(5, 3))
        value_counts = df[var].value_counts()
        labels = [f"{idx} ({count})" for idx, count in zip(value_counts.index, value_counts)]
        ax.pie(value_counts, labels=labels, autopct='%1.1f%%', colors=pie_colors[:len(value_counts)])
        ax.set_title(var)
        ax.axis('equal')
        cols[i % 2].pyplot(fig)
        cols[i % 2].caption(f"`{var}` → {get_chinese_name(var)}")  # 添加中文注释

    # =====================
    # c) 不同目标下的箱型图
    # =====================
    st.subheader("c) 是否患病 vs 各指标箱型图")
    cols = st.columns(2)
    for i, var in enumerate(continuous_vars):
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.boxplot(x='target', y=var, data=df, ax=ax, palette="Set2")
        ax.set_xlabel('target')
        ax.set_ylabel(var)
        cols[i % 2].pyplot(fig)
        cols[i % 2].caption(f"`{var}` → {get_chinese_name(var)}")  # 添加中文注释

    # =====================
    # d) 相关系数图（仅连续变量）
    # =====================
    st.subheader("d) 连续变量相关系数热力图")
    corr_df = df[continuous_vars].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_df, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax,
                xticklabels=continuous_vars,
                yticklabels=continuous_vars)
    st.pyplot(fig)
    # 在相关系数图下方添加匹配关系
    st.markdown("#### 字段说明")
    st.markdown("| 英文字段 | 中文含义 |")
    st.markdown("| --- | --- |")
    for var in continuous_vars:
        st.markdown(f"| `{var}` | {get_chinese_name(var)} |")
        
def render_prediction(model):
    st.title("🫀 心脏病概率预测")

    input_data = {}

    with st.form("prediction_form"):
        st.markdown("请填写以下健康指标以预测是否患有心脏病：")
        col1, col2 = st.columns(2)

        input_fields = {
            "age": "年龄（岁）",
            "sex": "性别（0=女，1=男）",
            "trestbps": "静息血压（mm Hg）",
            "chol": "血清胆固醇（mg/dl）",
            "fbs": "空腹血糖 > 120 mg/dl（0=否，1=是）",
            "thalach": "最大心率",
            "exang": "运动诱发心绞痛（0=否，1=是）",
            "thal": "地中海贫血类型（0=正常，1=固定缺陷，2=可逆缺陷）"
        }

        for i, (key, label) in enumerate(input_fields.items()):
            with col1 if i % 2 == 0 else col2:
                input_data[key] = st.number_input(label=label, value=0, step=1)

        submit_button = st.form_submit_button("预测")

    if submit_button:
        input_df = pd.DataFrame([input_data])
        proba = model.predict_proba(input_df)[0][1]
        st.success(f"预测患心脏病的概率为：**{proba * 100:.2f}%**")
def main():
    set_background_image('background.jpg')

    df = load_and_clean_data()
    model, _ = train_model(df)
    
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    page = sidebar_navigation()

    if page == "登录与注册":
        render_login_register()
    elif st.session_state['logged_in']:
        if page == "数据分析与可视化":
            render_visualizations(df)
        elif page == "心脏病预测":
            render_prediction(model)
        elif page == "个人信息":
            render_profile()
    else:
        st.warning("请先登录以访问此页面。")


if __name__ == "__main__":
    main()

