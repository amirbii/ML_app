import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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
from sklearn.metrics import classification_report, accuracy_score, consensus_score, confusion_matrix
from sklearn.neighbors import LocalOutlierFactor

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
####################
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

####################
if df is not None:
    st.subheader("اطلاعات دیتا 📊")

    st.write(f"🔢 شکل داده: {df.shape[0]} نمونه × {df.shape[1]} ستون")

    st.write("Data Types")
    st.write(df.dtypes)

    st.write("Missing Values")
    st.write(df.isnull().sum())

    st.write("Descriptive Statistics:")
    st.write(df.describe(include='all'))

    ####################
    st.subheader("حذف داده‌های پرت 🧹")

    out = st.radio(" روش های حذف داده", ["None", "STD + Mean", "IQR", "LOF"])
    button = st.button("🚀 اجرای حذف")

    num = df.select_dtypes(include=np.number).columns
    lenn = len(df)
    df_out = df.copy()
    x = df_out[num]

    if out == "None":
        st.session_state.df_out = df
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

            lof = LocalOutlierFactor(n_neighbors=3)
            outlier_pred = lof.fit_predict(x)
            outlier_index = np.where(outlier_pred == -1)
            df_out = df_out.drop(index=outlier_index[0])

        removed = lenn - len(df_out)
        percent = removed / lenn * 100
        st.success(f" نمونه حذف شدند: {removed}")
        st.markdown(f"**🎯 درصد نمونه های باقی مانده:** {percent:.2f}")
        st.session_state.df_out = df_out

if "df_out" in st.session_state:
    st.subheader("دیتای باقی مانده 📉")
    st.write(f"شکل داده: {df_out.shape[0]} نمونه × {df_out.shape[1]} ستون")
    st.write(df_out.describe())

####################
st.header("پیش پردازش 🧹")
scale_method = st.radio("روش نرمال‌سازی داده‌ها را انتخاب کنید:", ("None", "StandardScaler", "MinMaxScaler"))
button1 = st.button(" نرمال‌سازی")

if button1 and scale_method != "None":
    if 'df_out' in st.session_state:
        df_out = st.session_state.df_out
    else:
        df_out = df

    numeric_cols = df_out.select_dtypes(include=np.number).columns

    if scale_method == "StandardScaler":
        scaler = StandardScaler()
    elif scale_method == "MinMaxScaler":
        scaler = MinMaxScaler()

    scaled_array = scaler.fit_transform(df_out[numeric_cols])
    df_scaled = pd.DataFrame(scaled_array, columns=numeric_cols)

    st.subheader("داده‌های نرمال‌شده:")
    st.dataframe(df_scaled.head())

elif button1 and scale_method == "None":
    st.info("نرمال‌سازی انتخاب نشده است.")
####################
st.header("تقسیم داده ➗")

test_size = st.slider("مقدار تست", min_value=0.0, max_value=0.5, step=0.05, value=0.2)

col1, col2 = st.columns(2)
with col1:
    shuffle = st.checkbox("🔀 Shuffle", value=True)
with col2:
    stratify = st.checkbox("🎯 Stratify", value=False)

# انتخاب دیتا
if 'df_scaled' in st.session_state:
    df_final = st.session_state.df_scaled
elif 'df_out' in st.session_state:
    df_final = st.session_state.df_out
elif 'df' in locals() and df is not None:
    df_final = df
else:
    df_final = None

target_column = None
button2 = False

if df_final is not None and len(df_final) > 1:
    st.subheader("🎯 انتخاب ستون هدف:")
    target_column = st.selectbox("ستون لیبل (y):", df_final.columns)
    button2 = st.button("Train/Test Split")

if button2 and target_column is not None:
    X = df_final.drop(columns=[target_column])
    y = df_final[target_column]

    st.write("📊 تعداد نمونه در هر کلاس:")
    st.write(y.value_counts())

    if stratify and y.value_counts().min() < 2:
        st.error("❌ برای Stratify، هر کلاس باید حداقل ۲ نمونه داشته باشد.")
    else:
        stratify_value = y if stratify else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            shuffle=shuffle,
            stratify=stratify_value,
            random_state=42
        )

        # ذخیره در session
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test

        st.success("✅ داده‌ها با موفقیت تقسیم شدند.")
        st.write(f"🟩 آموزش: {X_train.shape[0]} نمونه")
        st.write(f"🟥 تست: {X_test.shape[0]} نمونه")


####################
st.title("انواع مدل 🤖")

model = st.selectbox("انتخاب مدل", ["Logistic", "SVM", "KNN", "Decision Tree"])

st.markdown("### پارامترها ⚙️")

if model == "Logistic":
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        penalty = st.selectbox("Penalty", ["l2", "none"])
    with col2:
        solver = st.selectbox("Solver", ["lbfgs", "liblinear", "saga"])
    with col3:
        C = st.number_input("C", 0.01, 10.0, value=1.0, step=0.1)
    with col4:
        max_iter = st.slider("Max Iterations", 100, 1000, 200, step=50)


elif model == "SVM":
    col1, col2, col3 = st.columns(3)
    with col1:
        c = st.number_input("C", min_value=0.01, max_value=100.0, value=1.0, step=0.1)
    with col2:
        kernel = st.selectbox("Kernel", ["linear", "rbf", "poly", "sigmoid"])
    with col3:
        gamma = st.selectbox("Gamma", ["auto", "scale"])


elif model == "KNN":
    col1, col2, col3 = st.columns(3)
    with col1:
        k = st.slider("K", min_value=1, max_value=50, value=5)
    with col2:
        weight = st.selectbox("Weight", ["uniform", "distance"])
    with col3:
        metric = st.selectbox("Metric", ["euclidean", "manhattan", "minkowski"])


elif model == "Decision Tree":
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        criterion = st.selectbox("Criterion", ["gini", "entropy"])
    with col2:
        max_depth = st.number_input("Max Depth", min_value=1, max_value=50, value=5)
    with col3:
        min_samples_leaf = st.number_input("min_samples_leaf", min_value=1, max_value=100, value=1)
    with col4:
        min_samples_split = st.number_input("min_samples_split", min_value=2, max_value=20, value=2)
####################
if st.button("Auto Tuning"):
    st.write("Grid Search")

####################
if st.button("Train"):
    if not all(k in st.session_state for k in ["X_train", "X_test", "y_train", "y_test"]):
        st.warning("⚠️ ابتدا داده‌ها را با Train/Test Split تقسیم کنید.")
    else:
        X_train = st.session_state.X_train
        y_train = st.session_state.y_train
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test

        if model == "Logistic":
            clf = LogisticRegression(penalty=penalty, solver=solver, C=C, max_iter=max_iter)
        elif model == "SVM":
            clf = SVC(C=c, kernel=kernel, gamma=gamma)
        elif model == "KNN":
            clf = KNeighborsClassifier(n_neighbors=k, weights=weight, metric=metric)
        elif model == "Decision Tree":
            clf = DecisionTreeClassifier(
                criterion=criterion,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                min_samples_split=min_samples_split
            )

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        st.success("✅ مدل با موفقیت آموزش داده شد")
        st.markdown(f"**🎯 دقت مدل:** {acc * 100:.2f}")


        st.subheader("📊 Confusion Matrix")
        st.write(confusion_matrix(y_test, y_pred))

        st.subheader("📋 Classification Report")
        st.text(classification_report(y_test, y_pred))

    ####################

####################
x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
y = [1, 3, 4, 5, 6]
fig = go.Figure(data=go.Scatter(x=x, y=y))
st.plotly_chart(fig)
####################
st.title("انواع مدل بوست 🤖")
####################
st.header("تست مدل 🖼️")
image_file = st.file_uploader("فایل تست آپلود کنید", type=["jpg", "jpeg", "png"])

####################
st.header("تولید کد نهایی 🧾")
if st.button("Generat Code"):
    pass
#####################
