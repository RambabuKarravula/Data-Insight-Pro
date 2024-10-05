# DataInsight Pro: Comprehensive Data Analysis and ML Dashboard
## Overview
DataInsight Pro is a powerful, interactive web application built with Streamlit that provides a comprehensive suite of data analysis and machine learning tools. This dashboard allows users to upload data, perform preprocessing, visualize insights, train machine learning models, make predictions, and compare model performances - all through an intuitive and user-friendly interface.
Features

## Data Upload and Preprocessing

CSV file upload functionality
Automatic removal of rows with NaN values
Data overview and summary statistics


## Data Visualization

Interactive charts including:

Scatter plots
Line charts
Bar charts
Box plots
Histograms
Heatmaps
3D scatter plots
Violin plots


Customizable axes and parameters


## Model Training and Evaluation

Support for multiple ML models:

Random Forest Regressor
Linear Regression
Support Vector Regression (SVR)
Random Forest Classifier
Logistic Regression
Support Vector Classification (SVC)


## Hyperparameter tuning
Model performance metrics (MSE, R-squared, Accuracy)
Feature importance visualization


## Prediction

Make predictions using trained models
Interactive input for feature values
Visualization of prediction results


## Model Comparison

Compare performance across different models
Bar chart visualization of model metrics


## Report Generation

Download trained models and scalers
Generate and download comprehensive PDF reports



## setup

Clone this repository:
Copy git clone https://github.com/RambabuKarravula/Data-Insight-Pro.git
cd datainsight-pro

Install the required packages:
Copypip install -r requirements.txt


## Usage

Run the Streamlit app:
Copy streamlit run app.py

Open your web browser and navigate to the provided local URL (usually http://localhost:8501).
Follow the on-screen instructions to upload your data, visualize it, train models, and generate insights.

## Requirements
See requirements.txt for a full list of dependencies.
Contributing
Contributions to DataInsight Pro are welcome! Please feel free to submit a Pull Request.
License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

Streamlit for the amazing framework
Plotly for interactive visualizations
scikit-learn for machine learning capabilities
