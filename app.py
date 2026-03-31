import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------- PAGE CONFIG -------------------
st.set_page_config(
    page_title="Air Pollution Dashboard",
    page_icon="🌍",
    layout="wide"
)

# ------------------- LOAD FILES -------------------
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

# ------------------- SIDEBAR -------------------
st.sidebar.title("🌍 Air Pollution App")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "📊 Data Analysis", "🤖 Prediction", "📈 Visualization", "ℹ️ About"]
)

# ------------------- HOME -------------------
if page == "🏠 Home":
    st.title("🌫️ Air Pollution Analysis Dashboard")

    st.markdown("""
    This project analyzes air pollution levels across Indian cities using Machine Learning.
    
    ### 🔍 Features:
    - Data Analysis
    - Pollution Visualization
    - Air Quality Prediction
    - Interactive Dashboard
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Dataset Size", "50,000 Rows")
    col2.metric("Features", "72 Columns")
    col3.metric("Model Accuracy", "99.52%")

    st.image("https://images.unsplash.com/photo-1581092335397-9583eb92d232", use_container_width=True)

# ------------------- DATA ANALYSIS -------------------
elif page == "📊 Data Analysis":
    st.title("📊 Dataset Overview")

    st.write("### 🔹 Raw Data")
    st.dataframe(df.head())

    st.write("### 🔹 Dataset Info")
    st.write(df.describe())

    st.write("### 🔹 Missing Values")
    st.write(df.isnull().sum())

# ------------------- PREDICTION -------------------
elif page == "🤖 Prediction":
    st.title("🤖 Air Quality Prediction")

    st.sidebar.subheader("Enter Feature Values")

    input_data = {}

    # Dynamic input fields
    for feature in features[:10]:  # limit for UI simplicity
        input_data[feature] = st.sidebar.number_input(feature, value=float(df[feature].mean()))

    input_df = pd.DataFrame([input_data])

    st.write("### 🔹 Input Data")
    st.dataframe(input_df)

    if st.button("Predict"):
        prediction = model.predict(input_df)[0]

        st.subheader("Prediction Result")

        # Color indicator
        if str(prediction).lower() == "good":
            st.success(f"🟢 {prediction} Air Quality")
        elif str(prediction).lower() == "moderate":
            st.warning(f"🟡 {prediction} Air Quality")
        else:
            st.error(f"🔴 {prediction} Air Quality")

# ------------------- VISUALIZATION -------------------
elif page == "📈 Visualization":
    st.title("📈 Data Visualization")

    st.subheader("Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("PM2.5 Distribution")

    if "PM2.5" in df.columns:
        fig2, ax2 = plt.subplots()
        sns.histplot(df["PM2.5"], kde=True, ax=ax2)
        st.pyplot(fig2)

    st.subheader("City-wise Pollution")

    if "City" in df.columns and "PM2.5" in df.columns:
        city_data = df.groupby("City")["PM2.5"].mean().sort_values(ascending=False).head(10)

        fig3, ax3 = plt.subplots()
        city_data.plot(kind="bar", ax=ax3)
        st.pyplot(fig3)

# ------------------- ABOUT -------------------
elif page == "ℹ️ About":
    st.title("ℹ️ About Project")

    st.markdown("""
    ### 🌍 Air Pollution Analysis in Indian Cities
    
    This project uses Machine Learning to analyze and predict air quality levels.
    
    ### 🛠 Technologies Used:
    - Python
    - Pandas, NumPy
    - Scikit-learn
    - Streamlit
    
    ### 📊 Model:
    - Random Forest Classifier
    - Accuracy: 99.52%
    
    ### 👩‍💻 Developed By:
    - Your Name
    
    ### 🎯 Objective:
    To predict air quality and help reduce pollution impact.
    """)

# ------------------- FOOTER -------------------
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")
