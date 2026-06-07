import gradio as gr
import pandas as pd
import numpy as np
import joblib
import json

# Load the machine learning models
lstm_model = joblib.load('data/saved_ml_models/lstm_pakistan_health_expenditure.h5')
with open('data/saved_ml_models/prophet_pakistan_health_expenditure.json') as f:
    prophet_model = json.load(f)

# Define prediction functions
def predict_health_expenditure(input_data):
    # Process input data and make predictions using the LSTM model
    # Placeholder for actual prediction logic
    prediction = lstm_model.predict(input_data)
    return prediction

def forecast_health_expenditure(input_data):
    # Process input data and make forecasts using the Prophet model
    # Placeholder for actual forecasting logic
    forecast = prophet_model.predict(input_data)
    return forecast

# Create Gradio interface
with gr.Blocks(css="assets/styles.css") as app:
    gr.Markdown("<h1 style='text-align: center;'>Global Life Expectancy Prediction</h1>")
    
    with gr.Row():
        with gr.Column():
            input_data = gr.Dataframe(label="Input Data", type="pandas")
            predict_button = gr.Button("Predict Health Expenditure")
            forecast_button = gr.Button("Forecast Health Expenditure")
        
        with gr.Column():
            prediction_output = gr.Textbox(label="Prediction Output")
            forecast_output = gr.Textbox(label="Forecast Output")
    
    predict_button.click(predict_health_expenditure, inputs=input_data, outputs=prediction_output)
    forecast_button.click(forecast_health_expenditure, inputs=input_data, outputs=forecast_output)

# Launch the app
app.launch()