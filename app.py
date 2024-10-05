import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from pydantic import BaseModel, Field
import pickle
import base64
from fpdf import FPDF
import io
from PIL import Image
from st_aggrid import AgGrid, GridOptionsBuilder

# Set page config
st.set_page_config(page_title="Comprehensive Data Analysis Dashboard", layout="wide")

# Custom CSS to enhance the visual appeal
st.markdown("""
<style>
    .reportview-container {
        background: #1E1E1E;
        color: #FFFFFF;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3, h4 {
        color: #FFFFFF;
    }
    .stButton>button {
        color: #FFFFFF;
        background-color: #4CAF50;
        border-radius: 5px;
    }
    .stTextInput>div>div>input {
        color: #FFFFFF;
        background-color: #2E2E2E;
        border-radius: 5px;
    }
    .stSelectbox>div>div>select {
        color: #FFFFFF;
        background-color: #2E2E2E;
    }
    .stExpander {
        background-color: #2E2E2E;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .css-145kmo2 {
        color: #FFFFFF;
    }
    .css-1d391kg {
        color: #FFFFFF;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'original_data' not in st.session_state:
    st.session_state.original_data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'models' not in st.session_state:
    st.session_state.models = {}

# Functions
@st.cache_data
def load_data(file):
    try:
        data = pd.read_csv(file)
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def get_column_types(data):
    return {
        'numeric': data.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical': data.select_dtypes(include=['object']).columns.tolist()
    }

def remove_nan(data):
    return data.dropna()

def train_model(X, y, model_type, params):
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        if model_type == "Random Forest Regressor":
            model = RandomForestRegressor(**params)
        elif model_type == "Linear Regression":
            model = LinearRegression(**params)
        elif model_type == "SVR":
            model = SVR(**params)
        elif model_type == "Random Forest Classifier":
            model = RandomForestClassifier(**params)
        elif model_type == "Logistic Regression":
            model = LogisticRegression(**params)
        else:  # SVC
            model = SVC(**params)
        
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        if model_type in ["Random Forest Regressor", "Linear Regression", "SVR"]:
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            return model, scaler, mse, r2, None
        else:
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            return model, scaler, accuracy, None, report
    except Exception as e:
        st.error(f"Error during model training: {str(e)}")
        return None, None, None, None, None

def save_model(model, scaler):
    model_pickle = pickle.dumps(model)
    scaler_pickle = pickle.dumps(scaler)
    return model_pickle, scaler_pickle

def create_download_link(val, filename):
    b64 = base64.b64encode(val)
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}">Download file</a>'

def create_pdf(data, visualizations, model_results):
    pdf = FPDF()
    pdf.add_page()
    
    # Add title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Data Analysis Report", ln=True, align="C")
    pdf.ln(10)
    
    # Add data overview
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Data Overview", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, f"Number of rows: {data.shape[0]}\nNumber of columns: {data.shape[1]}")
    pdf.ln(10)
    
    # Add visualizations
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Visualizations", ln=True)
    for title, img_bytes in visualizations.items():
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, title, ln=True)
        img = Image.open(img_bytes)
        img_path = f"temp_{title.replace(' ', '_')}.png"
        img.save(img_path)
        pdf.image(img_path, x=10, w=190)
        pdf.ln(10)
    
    # Add model results
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Model Results", ln=True)
    pdf.set_font("Arial", "", 12)
    for key, value in model_results.items():
        pdf.multi_cell(0, 10, f"{key}: {value}")
    
    return pdf.output(dest="S").encode("latin-1")

# Pydantic model for config
class AppConfig(BaseModel):
    title: str = Field("Comprehensive Data Analysis Dashboard", description="The title of the app")
    max_file_size: int = Field(200, description="Maximum file size in MB")

# Create an instance of the config
config = AppConfig()

# Main App
st.title(config.title)

# Data Upload and Preprocessing Section
st.header("1. Data Upload and Preprocessing")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    file_size = uploaded_file.size / (1024 * 1024)  # Convert to MB
    if file_size > config.max_file_size:
        st.error(f"File size exceeds the maximum limit of {config.max_file_size}MB.")
    else:
        data = load_data(uploaded_file)
        if data is not None:
            st.session_state.original_data = data.copy()
            st.success("Data loaded successfully!")
            
            # Remove rows with any null values
            data = remove_nan(data)
            st.session_state.data = data
            
            with st.expander("View Data Sample"):
                st.write(data.head())
            
            with st.expander("View Data Overview"):
                st.write("Data Shape:", data.shape)
                st.write("Column Info:")
                st.write(data.dtypes)
                st.write("Summary Statistics:")
                st.write(data.describe())
            
            st.success(f"Rows with NaN values have been removed. New data shape: {data.shape}")

# Data Visualization Section
if st.session_state.data is not None:
    st.header("2. Data Visualization")
    column_types = get_column_types(st.session_state.data)
    
    chart_types = ["Scatter Plot", "Line Chart", "Bar Chart", "Box Plot", "Histogram", "Heatmap", "3D Scatter", "Violin Plot"]
    visualizations = {}
    
    for chart_type in chart_types:
        st.subheader(chart_type)
        col1, col2 = st.columns(2)
        with col1:
            x_column = st.selectbox(f"Select X-axis for {chart_type}", column_types['numeric'] + column_types['categorical'], key=f"x_{chart_type}")
        with col2:
            y_column = st.selectbox(f"Select Y-axis for {chart_type}", column_types['numeric'], key=f"y_{chart_type}")
        
        if chart_type == "Scatter Plot":
            fig = px.scatter(st.session_state.data, x=x_column, y=y_column, title=f"{x_column} vs {y_column}")
        elif chart_type == "Line Chart":
            fig = px.line(st.session_state.data, x=x_column, y=y_column, title=f"{x_column} vs {y_column}")
        elif chart_type == "Bar Chart":
            fig = px.bar(st.session_state.data, x=x_column, y=y_column, title=f"{x_column} vs {y_column}")
        elif chart_type == "Box Plot":
            fig = px.box(st.session_state.data, x=x_column, y=y_column, title=f"{x_column} vs {y_column}")
        elif chart_type == "Histogram":
            fig = px.histogram(st.session_state.data, x=x_column, title=f"Histogram of {x_column}")
        elif chart_type == "Heatmap":
            corr = st.session_state.data[column_types['numeric']].corr()
            fig = px.imshow(corr, title="Correlation Heatmap")
        elif chart_type == "3D Scatter":
            z_column = st.selectbox(f"Select Z-axis for {chart_type}", column_types['numeric'], key=f"z_{chart_type}")
            fig = px.scatter_3d(st.session_state.data, x=x_column, y=y_column, z=z_column, title=f"3D Scatter: {x_column} vs {y_column} vs {z_column}")
        else:  # Violin Plot
            fig = px.violin(st.session_state.data, x=x_column, y=y_column, title=f"Violin Plot: {x_column} vs {y_column}")
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#FFFFFF'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Save the plot for PDF
        img_bytes = fig.to_image(format="png")
        visualizations[chart_type] = io.BytesIO(img_bytes)

# Model Training Section
if st.session_state.data is not None:
    st.header("3. Model Training and Evaluation")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        target_column = st.selectbox("Select target column", column_types['numeric'])
    with col2:
        features = st.multiselect("Select features", column_types['numeric'], default=column_types['numeric'])
        st.session_state.features = features
    with col3:
        model_type = st.selectbox("Select model type", ["Random Forest Regressor", "Linear Regression", "SVR", "Random Forest Classifier", "Logistic Regression", "SVC"])
    
    # Model parameters
    st.subheader("Model Parameters")
    if model_type == "Random Forest Regressor" or model_type == "Random Forest Classifier":
        n_estimators = st.slider("Number of estimators", 10, 200, 100)
        max_depth = st.slider("Max depth", 1, 20, 5)
        params = {"n_estimators": n_estimators, "max_depth": max_depth}
    elif model_type == "Linear Regression":
        params = {}  # Linear Regression doesn't have hyperparameters to tune
    elif model_type == "SVR" or model_type == "SVC":
        C = st.slider("C (Regularization parameter)", 0.01, 10.0, 1.0)
        kernel = st.selectbox("Kernel", ["rbf", "linear", "poly"])
        params = {"C": C, "kernel": kernel}
    else:  # Logistic Regression
        C = st.slider("C (Regularization parameter)", 0.01, 10.0, 1.0)
        params = {"C": C}
    
    if st.button("Train Model", key="train_model"):
        X = st.session_state.data[features]
        y = st.session_state.data[target_column]
        
        with st.spinner("Training model..."):
            model, scaler, metric, r2, report = train_model(X, y, model_type, params)
            if model is not None:
                st.session_state.model = model
                st.session_state.scaler = scaler
                st.success("Model trained successfully!")
                
                model_results = {}
                if model_type in ["Random Forest Regressor", "Linear Regression", "SVR"]:
                    st.metric("Mean Squared Error", f"{metric:.4f}")
                    st.metric("R-squared Score", f"{r2:.4f}")
                    model_results["Mean Squared Error"] = f"{metric:.4f}"
                    model_results["R-squared Score"] = f"{r2:.4f}"
                else:
                    st.metric("Accuracy", f"{metric:.4f}")
                    st.text("Classification Report:")
                    st.text(report)
                    model_results["Accuracy"] = f"{metric:.4f}"
                    model_results["Classification Report"] = report

                if hasattr(model, 'feature_importances_'):
                    importance = model.feature_importances_
                    feature_importance = pd.DataFrame({'Feature': features, 'Importance': importance})
                    feature_importance = feature_importance.sort_values('Importance', ascending=False).reset_index(drop=True)
                    st.subheader("Feature Importance")
                    # Use Streamlit's native table display
                    st.table(feature_importance)
                    # Save feature importance as a table for PDF
                    fig_importance = go.Figure(data=[go.Table(
                        header=dict(values=list(feature_importance.columns),
                                    fill_color='paleturquoise',
                                    align='left'),
                        cells=dict(values=[feature_importance.Feature, feature_importance.Importance],
                                   fill_color='lavender',align='left'))
                                                     ])
                    img_bytes = fig_importance.to_image(format="png")
                    visualizations["Feature Importance"] = io.BytesIO(img_bytes)

                # Save model and scaler
                model_pickle, scaler_pickle = save_model(model, scaler)
                st.markdown(create_download_link(model_pickle, "trained_model.pkl"), unsafe_allow_html=True)
                st.markdown(create_download_link(scaler_pickle, "scaler.pkl"), unsafe_allow_html=True)

                # Generate PDF report
                pdf_buffer = create_pdf(st.session_state.data, visualizations, model_results)
                st.markdown(create_download_link(pdf_buffer, "data_analysis_report.pdf"), unsafe_allow_html=True)

                # Store model info for comparison
                st.session_state.models[model_type] = {
                    'model': model,
                    'scaler': scaler,
                    'metric': metric,
                    'r2': r2
                }

# Prediction Section
if st.session_state.model is not None and st.session_state.scaler is not None:
    st.header("4. Make Predictions")
    
    with st.expander("Enter values for prediction"):
        input_data = {}
        col1, col2 = st.columns(2)
        for i, feature in enumerate(st.session_state.features):
            with col1 if i % 2 == 0 else col2:
                input_data[feature] = st.number_input(f"Enter value for {feature}", value=float(st.session_state.data[feature].mean()))
        
        if st.button("Predict", key="make_prediction"):
            input_df = pd.DataFrame([input_data])
            input_scaled = st.session_state.scaler.transform(input_df)
            prediction = st.session_state.model.predict(input_scaled)
            st.success(f"Predicted value: {prediction[0]:.4f}")

            # Visualize prediction
            fig = go.Figure()
            fig.add_trace(go.Indicator(
                mode="number+delta",
                value=prediction[0],
                delta={'reference': st.session_state.data[target_column].mean(), 'relative': True},
                title={'text': "Prediction vs Mean"},
                domain={'x': [0, 1], 'y': [0, 1]}
            ))
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#FFFFFF'
            )
            st.plotly_chart(fig, use_container_width=True)

# Model Comparison
if len(st.session_state.models) > 1:
    st.header("5. Model Comparison")
    
    comparison_data = []
    for model_name, model_info in st.session_state.models.items():
        if model_info['r2'] is not None:
            comparison_data.append({
                'Model': model_name,
                'Metric': 'R-squared',
                'Value': model_info['r2']
            })
        comparison_data.append({
            'Model': model_name,
            'Metric': 'MSE' if model_info['r2'] is not None else 'Accuracy',
            'Value': model_info['metric']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    fig = px.bar(comparison_df, x='Model', y='Value', color='Metric', barmode='group',
                 title='Model Comparison')
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#FFFFFF'
    )
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.info("Comprehensive Data Analysis Dashboard v3.0")

if __name__ == "__main__":
    st.sidebar.title("About")
    st.sidebar.info(
        "This dashboard allows you to upload data, preprocess it, visualize it, "
        "train machine learning models, make predictions, and compare different models. "
        "You can also download the trained model, scaler, and a PDF report of your analysis."
    )
    st.sidebar.title("Instructions")
    st.sidebar.info(
        "1. Upload your CSV file\n"
        "2. NaN values will be automatically removed\n"
        "3. Explore the data visualizations\n"
        "4. Train a model by selecting features and model type\n"
        "5. Make predictions using the trained model\n"
        "6. Compare different models (if multiple models are trained)\n"
        "7. Download the trained model, scaler, and PDF report"
    )