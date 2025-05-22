import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import shap

# Set page config (must be first Streamlit command)
st.set_page_config(page_title="💼 Salary Predictor Pro", page_icon="💰", layout="wide")

# Load model
model = joblib.load("linearmodel.pkl")  # Make sure this file exists in your working directory

# Prepare background data for SHAP (replace with your training data if you have it)
background_data = np.array([
    [5, 3.5],
    [10, 4.0],
    [2, 2.5],
])

# Initialize SHAP explainer with masker
explainer = shap.Explainer(model.predict, masker=background_data, feature_names=['years', 'jobrate'])

# Title and description
st.title("💼 AI-Powered Salary Prediction App")
st.markdown("Use this tool to predict employee salaries based on experience and performance.")

st.divider()

# Tabs for Single Prediction, Batch Prediction, Analytics
tab1, tab2, tab3 = st.tabs(["🔮 Predict One", "📂 Batch Prediction", "📈 Analytics"])

# ---------- TAB 1: Single Prediction ----------
with tab1:
    st.header("🔍 Single Employee Prediction")

    col1, col2 = st.columns(2)

    with col1:
        years = st.slider("Years at Company", min_value=0, max_value=40, value=1)
        jobrate = st.slider("Job Performance Rating", min_value=1.0, max_value=5.0, step=0.1, value=3.5)

        if st.button("💰 Predict Salary"):
            x = np.array([[years, jobrate]])
            prediction = model.predict(x)[0]
            st.success(f"Predicted Salary: ${prediction:,.2f}")

            # SHAP explanation for this input
            shap_values = explainer(x)
            st.subheader("🔎 Model Explanation (SHAP)")
            st_shap_html = shap.plots.waterfall(shap_values[0], show=False)
            # Render SHAP waterfall plot using streamlit.pyplot or via HTML
            import matplotlib.pyplot as plt
            fig = plt.gcf()
            st.pyplot(fig)

    with col2:
        st.markdown("**Input Summary:**")
        st.write(f"- Years at company: {years}")
        st.write(f"- Job performance rating: {jobrate}")

# ---------- TAB 2: Batch Prediction ----------
with tab2:
    st.header("📂 Batch Salary Predictions")

    uploaded_file = st.file_uploader("Upload CSV file with columns: 'years', 'jobrate'", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:")
        st.dataframe(df)

        if st.button("Predict Salaries for Batch"):
            if 'years' in df.columns and 'jobrate' in df.columns:
                X_batch = df[['years', 'jobrate']].values
                preds = model.predict(X_batch)
                df['predicted_salary'] = preds
                st.success("Prediction completed!")
                st.dataframe(df)

                # Download updated dataframe as CSV
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(label="📥 Download Predictions CSV", data=csv, file_name='salary_predictions.csv', mime='text/csv')

                # Plot salary distribution
                fig = px.histogram(df, x='predicted_salary', nbins=30, title='Predicted Salary Distribution')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("CSV must have 'years' and 'jobrate' columns.")

# ---------- TAB 3: Analytics ----------
with tab3:
    st.header("📈 Data & Model Analytics")

    st.markdown("Upload a CSV to explore data and get SHAP summary.")

    analytics_file = st.file_uploader("Upload CSV with 'years' and 'jobrate'", key="analytics")
    if analytics_file is not None:
        df_analytics = pd.read_csv(analytics_file)
        st.write("Data Preview:")
        st.dataframe(df_analytics.head())

        if 'years' in df_analytics.columns and 'jobrate' in df_analytics.columns:
            X_analytics = df_analytics[['years', 'jobrate']].values
            preds = model.predict(X_analytics)
            df_analytics['predicted_salary'] = preds

            st.markdown("### Predicted Salary Distribution")
            fig = px.histogram(df_analytics, x='predicted_salary', nbins=30)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### SHAP Summary Plot")
            shap_values_batch = explainer(X_analytics)
            fig_shap = shap.plots.beeswarm(shap_values_batch, show=False)
            import matplotlib.pyplot as plt
            fig = plt.gcf()
            st.pyplot(fig)

        else:
            st.error("CSV must contain 'years' and 'jobrate' columns.")

# ---------- Custom CSS Styling ----------
st.markdown("""
    <style>
    .css-1d391kg {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)
