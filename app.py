import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from PIL import Image
import os
import subprocess
from zipfile import ZipFile
import plotly.graph_objects as go
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

api_k = {"username": "amirbi",
         "key": "cce234fe761dad172e451eb0141f1143"}

# def apply_inline_styles():
# css = """
# * {
#  direction: rtl;
#     .st-emotion-cache-13ln4jf{
# max-width: 100% !important;.
# }
# }"""

# st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)


# apply_inline_styles()
#####
st.title("بارگذاری دیتاست 📁")

method = st.radio("روش بارگذاری:", ["📤 CSV", "🌐Github", "🌐kaggle"])

df = None

if method == "📤 CSV":
    uploaded_file = st.file_uploader("فایل CSV را انتخاب کنید", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("✅ فایل با موفقیت بارگذاری شد")

elif method == "🌐Github":
    url = st.text_input("link")
    if st.button("📥 بارگذاری"):
        try:
            df = pd.read_csv(url)
            st.success("✅ فایل با موفقیت بارگذاری شد")
        except Exception as e:
            st.error(f"❌ خطا در بارگذاری فایل: {e}")
elif method == "🌐kaggle":
    url = st.text_input("link")
    if st.button("📥 بارگذاری"):
        try:
            os.environ['KAGGLE_CONFIG_DIR'] = os.getcwd()

            parts = url.strip("/").split("/")
            slug = f"{parts[-2]}/{parts[-1]}"
            zip_name = f"{parts[-1]}.zip"

            subprocess.run(["kaggle", "datasets", "download", "-d", slug], check=True)

            with ZipFile(zip_name, 'r') as zip_ref:
                zip_ref.extractall("kaggle_data")

            for file in os.listdir("kaggle_data"):
                if file.endswith(".csv"):
                    df = pd.read_csv(os.path.join("kaggle_data", file))
                    st.success("✅ فایل CSV با موفقیت بارگذاری شد")
                    break
            else:
                st.warning("⚠️ فایل CSV در دیتاست یافت نشد.")
        except Exception as e:
            st.error(f"❌ خطا در بارگذاری از Kaggle: {e}")

if df is not None:
    st.subheader("اطلاعات دیتا 📊")

    st.write(f"🔢 شکل داده: {df.shape[0]} نمونه × {df.shape[1]} ستون")

    st.write("Data Types")
    st.write(df.dtypes)

    st.write("Missing Values")
    st.write(df.isnull().sum())

    st.write("Descriptive Statistics:")
    st.write(df.describe(include='all'))
    #####

    st.subheader("حذف داده‌های پرت 🧹")

    out = st.radio(" روش های حذف داده", ["None", "STD + Mean", "IQR", "LOF"])
    button = st.button("🚀 اجرای حذف")

    num = df.select_dtypes(include=np.number).columns
    lenn = len(df)
    df_out = df.copy()
    x = df_out[num]

    if out == "None":
        st.info("هیچ داده‌ای حذف نشده است")

    if out != "None" and button:
        if out == "STD + Mean":
            std = 1.5
            mean = x.mean()
            std_val = x.std()

            upper_bound = mean + std * std_val
            lower_bound = mean - std * std_val

            outlier_mask = ((x > upper_bound) | (x < lower_bound))
            valid_mask = ~outlier_mask.any(axis=1)
            df_out = df_out[valid_mask]

        elif out == "IQR":
            Q1 = x.quantile(0.25)
            Q3 = x.quantile(0.75)
            IQR = Q3 - Q1
            mask = ~((x < (Q1 - 1.5 * IQR)) | (x > (Q3 + 1.5 * IQR))).any(axis=1)
            df_out = df_out[mask]

        elif out == "LOF":
            from sklearn.neighbors import LocalOutlierFactor

            lof = LocalOutlierFactor(n_neighbors=3)
            outlier_pred = lof.fit_predict(x)
            outlier_index = np.where(outlier_pred == -1)
            df_out = df_out.drop(index=outlier_index[0])

        removed = lenn - len(df_out)
        percent = removed / lenn * 100
        st.success(f"✅ {removed} ردیف حذف شدند ({percent:.2f}٪)")

        st.subheader("دیتای باقی مانده 📉")
        st.write(f"شکل داده: {df_out.shape[0]} نمونه × {df_out.shape[1]} ستون")
        st.write(df_out.describe())

######
st.header("پیش پردازش 🧹")
scale_method = st.radio("روش نرمال‌سازی داده‌ها را انتخاب کنید:", ("None", "StandardScaler", "MinMaxScaler"))
######
st.header("تقسیم داده ➗")
test_size = st.slider("مقدار تست", min_value=0.0, max_value=0.5, step=0.1)
st.button("Train test split")
######
st.title("انواع مدل 🤖")

model = st.selectbox("انتخاب مدل", ["Logistic", "SVM", "KNN", "Decision Tree"])

if model == "Logistic":
    st.markdown("### پارامترها ⚙️")
    penalty = st.selectbox("Penalty", ["l2", "none"])
    solver = st.selectbox("Solver", ["lbfgs", "liblinear", "saga"])

elif model == "SVM":
    st.markdown("### پارامترها ⚙️")
    c = st.number_input("C (Regularization parameter)", min_value=0.01, max_value=100.0, value=1.0, step=0.1)
    kernel = st.selectbox("Kernel", ["linear", "rbf", "poly", "sigmoid"])

elif model == "KNN":
    st.markdown("### پارامترها ⚙️")
    k = st.number_input("K (تعداد همسایه‌ها)", min_value=1, max_value=50, value=5)

elif model == "Decision Tree":
    st.markdown("### پارامترها ⚙️")
    col1, col2, col3 = st.columns(3)

    with col1:
        criterion = st.selectbox("Criterion", ["gini", "entropy"])

    with col2:
        max_depth = st.number_input("Max Depth", min_value=1, max_value=100, value=5)

    with col3:
        min_samples_leaf = st.number_input("min_samples_leaf", min_value=2, max_value=100, value=5)

if st.button("Train"):
    st.write("Model Training")
    ####
    if st.button("Auto Tuning"):
        st.write("Grid Search")
x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
y = [1, 3, 4, 5, 6]
fig = go.Figure(data=go.Scatter(x=x, y=y))
st.plotly_chart(fig)
###
st.header("تست مدل 🖼️")
image_file = st.file_uploader("فایل تست آپلود کنید", type=["jpg", "jpeg", "png"])

###
st.header("تولید کد نهایی 🧾")
if st.button("Generat Code"):
    pass
