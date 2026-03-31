import streamlit as st
import pandas as pd
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Air Pollution Dashboard",
    page_icon="🌍",
    layout="wide"
)

# ---------------- PATH SETUP ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------- LOAD FILES ----------------
@st.cache_resource
def load_model():
    try:
        return joblib.load(os.path.join(BASE_DIR, "air_pol_pipe_model.pkl"))
    except:
        st.error("❌ Model file (air_pol_pipe_model.pkl) not found")
        st.stop()

@st.cache_data
def load_data():
    try:
        return pd.read_csv(os.path.join(BASE_DIR, "50k.csv"))
    except:
        st.error("❌ Dataset (50k.csv) not found")
        st.stop()

@st.cache_data
def load_features():
    try:
        with open(os.path.join(BASE_DIR, "features.json"), "r") as f:
            return json.load(f)
    except:
        st.error("❌ features.json not found")
        st.stop()

model = load_model()
df = load_data()
features = load_features()

# ---------------- STYLE ----------------
st.markdown("""
    <style>
    .main {
        background-color: #f4f6f7;
    }
    .stButton>button {
        background-color: #2E86C1;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🌍 Navigation")
page = st.sidebar.radio("", ["🏠 Dashboard", "📊 Analysis", "🤖 Prediction", "📈 Visualization", "ℹ️ About"])

# ---------------- DASHBOARD ----------------
if page == "🏠 Dashboard":
    st.title("🌫️ Air Pollution Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("📊 Total Records", len(df))
    col2.metric("📌 Features", df.shape[1])
    col3.metric("🤖 Model Accuracy", "99.52%")

    st.markdown("### 🌍 Project Overview")
    st.info("This dashboard analyzes air pollution and predicts air quality using Machine Learning.")

# ---------------- ANALYSIS ----------------
elif page == "📊 Analysis":
    st.title("📊 Data Analysis")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Statistical Summary")
    st.write(df.describe())

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

# ---------------- PREDICTION ----------------
elif page == "🤖 Prediction":
    st.title("🤖 Air Quality Prediction")

    st.info("Enter pollutant values to predict air quality")

    cols = st.columns(3)
    input_data = {}

    for i, feature in enumerate(features):
        if feature in df.columns:
            mean_val = float(df[feature].mean())

            if i % 3 == 0:
                input_data[feature] = cols[0].number_input(feature, value=mean_val)
            elif i % 3 == 1:
                input_data[feature] = cols[1].number_input(feature, value=mean_val)
            else:
                input_data[feature] = cols[2].number_input(feature, value=mean_val)

    input_df = pd.DataFrame([input_data])

    st.write("### 🔹 Input Data")
    st.dataframe(input_df)

    if st.button("🔍 Predict Air Quality"):
        try:
            # Fix input structure
            input_df = input_df.reindex(columns=features, fill_value=0)
            input_df = input_df.astype(float)

            result = model.predict(input_df)[0]

            st.subheader("🎯 Prediction Result")

            if str(result).lower() == "good":
                st.success(f"🟢 {result} Air Quality")
            elif str(result).lower() == "moderate":
                st.warning(f"🟡 {result} Air Quality")
            else:
                st.error(f"🔴 {result} Air Quality")

        except Exception as e:
            st.error("❌ Prediction failed")
            st.write(e)

# ---------------- VISUALIZATION ----------------
elif page == "📈 Visualization":
    st.title("📈 Data Visualization")

    numeric_df = df.select_dtypes(include=['number'])

    # Heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # PM2.5 Distribution
    if "PM2.5" in df.columns:
        st.subheader("PM2.5 Distribution")
        fig2, ax2 = plt.subplots()
        sns.histplot(df["PM2.5"], kde=True, ax=ax2)
        st.pyplot(fig2)

    # Top polluted cities
    if "City" in df.columns and "PM2.5" in df.columns:
        st.subheader("Top Polluted Cities")

        city_data = df.groupby("City")["PM2.5"].mean().sort_values(ascending=False).head(10)

        fig3, ax3 = plt.subplots()
        city_data.plot(kind="bar", ax=ax3)
        st.pyplot(fig3)

# ---------------- ABOUT ----------------
elif page == "ℹ️ About":
    st.title("ℹ️ About Project")

    st.markdown("""
    ### 🌍 Air Pollution Analysis in Indian Cities

    This project uses Machine Learning to analyze and predict air quality.

    **Model:** Random Forest  
    **Accuracy:** 99.52%  

    ### 🎯 Objective:
    To monitor and reduce air pollution using data-driven insights.

    ### 👩‍💻 Developed By:
    Priyanshi and vidisha
    """)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")
