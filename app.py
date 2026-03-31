import streamlit as st
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Air Quality Dashboard",
    page_icon="🌍",
    layout="wide"
)

# ---------------- LOAD ----------------
@st.cache_resource
def load_model():
    return joblib.load("air_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("50k.csv")

@st.cache_data
def load_features():
    with open("features.json", "r") as f:
        return json.load(f)

model = load_model()
df = load_data()
features = load_features()

# ---------------- STYLE ----------------
st.markdown("""
    <style>
    .main {background-color: #f5f7fa;}
    .stButton>button {
        background-color: #2E86C1;
        color: white;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🌍 Navigation")
page = st.sidebar.radio("", ["🏠 Dashboard", "📊 Analysis", "🤖 Prediction", "📈 Charts", "ℹ️ About"])

# ---------------- DASHBOARD ----------------
if page == "🏠 Dashboard":
    st.title("🌫️ Air Pollution Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", len(df))
    col2.metric("Total Features", df.shape[1])
    col3.metric("Model Accuracy", "99.52%")

    st.markdown("### 🌍 Overview")
    st.write("This dashboard analyzes air pollution and predicts air quality.")

# ---------------- ANALYSIS ----------------
elif page == "📊 Analysis":
    st.title("📊 Data Analysis")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Statistics")
    st.write(df.describe())

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

# ---------------- PREDICTION ----------------
elif page == "🤖 Prediction":
    st.title("🤖 Air Quality Prediction")

    st.markdown("Enter pollutant values:")

    col1, col2, col3 = st.columns(3)

    inputs = {}

    for i, feature in enumerate(features):
        if feature in df.columns:
            if i % 3 == 0:
                inputs[feature] = col1.number_input(feature, float(df[feature].mean()))
            elif i % 3 == 1:
                inputs[feature] = col2.number_input(feature, float(df[feature].mean()))
            else:
                inputs[feature] = col3.number_input(feature, float(df[feature].mean()))

    input_df = pd.DataFrame([inputs])

    st.write("### Input Data")
    st.dataframe(input_df)

    if st.button("🔍 Predict"):
        result = model.predict(input_df)[0]

        st.subheader("Prediction Result")

        if str(result).lower() == "good":
            st.success(f"🟢 {result} Air Quality")
        elif str(result).lower() == "moderate":
            st.warning(f"🟡 {result} Air Quality")
        else:
            st.error(f"🔴 {result} Air Quality")

# ---------------- CHARTS ----------------
elif page == "📈 Charts":
    st.title("📈 Visualizations")

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(df.corr(), cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    if "PM2.5" in df.columns:
        st.subheader("PM2.5 Distribution")
        fig2, ax2 = plt.subplots()
        sns.histplot(df["PM2.5"], kde=True, ax=ax2)
        st.pyplot(fig2)

# ---------------- ABOUT ----------------
elif page == "ℹ️ About":
    st.title("ℹ️ About Project")

    st.markdown("""
    ### 🌍 Air Pollution Analysis
    
    This project predicts air quality using machine learning.
    
    **Model Used:** Random Forest  
    **Accuracy:** 99.52%  
    
    ### 🎯 Goal
    Help monitor and reduce pollution levels.
    """)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")
