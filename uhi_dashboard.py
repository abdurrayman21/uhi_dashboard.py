# uhi_dashboard.py

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Set Streamlit page config
st.set_page_config(layout="wide", page_title="Urban Heat Island Dashboard")
st.title("🌆 Urban Heat Island Mitigation Dashboard")

# Upload CSV file
uploaded_file = st.file_uploader("📂 Upload the Urban Heat Island CSV Dataset", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        required_columns = [
            'City Name', 'Temperature (°C)', 'Population Density (people/km²)',
            'Land Cover', 'Urban Greenness Ratio (%)', 'Air Quality Index (AQI)',
            'Energy Consumption (kWh)'
        ]
        if not all(col in df.columns for col in required_columns):
            st.error("The uploaded CSV does not contain all the required columns.")
        else:
            # Data cleaning and preprocessing
            df['Land Cover'] = df['Land Cover'].astype('category').cat.codes
            df.dropna(inplace=True)

            # ---------------------- SIDEBAR ------------------------
            st.sidebar.header("🛠️ Visualization Settings")
            show_raw = st.sidebar.checkbox("Show Raw Data", False)

            if show_raw:
                st.subheader("🗃 Raw Dataset")
                st.write(df.head())

            # ------------------ EXPLORATORY ANALYSIS ------------------
            st.subheader("📊 Exploratory Visualizations")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Urban Temperature Distribution**")
                plt.figure(figsize=(6, 4))
                sns.histplot(df['Temperature (°C)'], kde=True)
                st.pyplot(plt.gcf())
                plt.clf()

            with col2:
                st.markdown("**Greenness Ratio vs Temperature**")
                plt.figure(figsize=(6, 4))
                sns.scatterplot(data=df, x='Urban Greenness Ratio (%)', y='Temperature (°C)')
                st.pyplot(plt.gcf())
                plt.clf()

            # ------------------ MACHINE LEARNING ----------------------
            st.subheader("🤖 AI-Based Analysis")

            # Linear Regression
            st.markdown("**Linear Regression: Predicting Urban Temperature**")
            features = [
                'Population Density (people/km²)', 'Land Cover',
                'Urban Greenness Ratio (%)', 'Air Quality Index (AQI)',
                'Energy Consumption (kWh)'
            ]
            X = df[features]
            y = df['Temperature (°C)']
            model = LinearRegression()
            model.fit(X, y)
            df['Predicted Temp (°C)'] = model.predict(X)

            st.write(df[['City Name', 'Temperature (°C)', 'Predicted Temp (°C)']].head())

            # KMeans Clustering
            st.markdown("**KMeans Clustering: UHI Risk Grouping**")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            df['Risk Cluster'] = kmeans.fit_predict(X_scaled)

            plt.figure(figsize=(6, 4))
            sns.scatterplot(
                data=df, x='Urban Greenness Ratio (%)', y='Temperature (°C)',
                hue='Risk Cluster', palette='coolwarm'
            )
            st.pyplot(plt.gcf())
            plt.clf()

            # ------------------ RECOMMENDATIONS ----------------------
            st.subheader("📌 City-wise Recommendations")

            def generate_recommendations(row):
                if row['Urban Greenness Ratio (%)'] < 25 and row['Temperature (°C)'] > 30:
                    return "🌳 Increase green space and reduce urban heat sources"
                elif row['Air Quality Index (AQI)'] > 150:
                    return "🚫 Control air pollution and promote cleaner transport"
                elif row['Population Density (people/km²)'] > 8000:
                    return "🏙️ Implement green roofs and vertical gardens"
                else:
                    return "✅ Conditions are moderate"

            df['Recommendation'] = df.apply(generate_recommendations, axis=1)

            st.dataframe(df[[
                'City Name', 'Temperature (°C)', 'Urban Greenness Ratio (%)',
                'Air Quality Index (AQI)', 'Recommendation'
            ]])

    except Exception as e:
        st.error(f"Something went wrong while processing the file: {e}")
else:
    st.info("👆 Please upload a CSV file to begin.")
