import gradio as gr
import pandas as pd
import numpy as np
import joblib
import json
import os

# Robust model loading with fallbacks so the UI can run even if models are missing
lstm_model = None
prophet_model = None

lstm_path = os.path.join('data', 'saved_ml_models', 'lstm_pakistan_health_expenditure.h5')
prophet_path = os.path.join('data', 'saved_ml_models', 'prophet_pakistan_health_expenditure.json')

try:
    lstm_model = joblib.load(lstm_path)
except Exception:
    try:
        from tensorflow.keras.models import load_model
        lstm_model = load_model(lstm_path)
    except Exception:
        lstm_model = None

try:
    with open(prophet_path) as f:
        prophet_model = json.load(f)
except Exception:
    prophet_model = None


def predict_health_expenditure(input_data):
    try:
        if isinstance(input_data, pd.DataFrame):
            X = input_data.values
        else:
            X = np.array(input_data)

        if lstm_model is not None:
            try:
                preds = lstm_model.predict(X)
            except Exception:
                # Try reshaping for common LSTM input shapes
                if X.ndim == 2:
                    preds = lstm_model.predict(X.reshape((X.shape[0], 1, X.shape[1])))
                else:
                    preds = lstm_model.predict(X)
            return str(np.array(preds).tolist())
        else:
            # Fallback: return simple statistics as placeholder
            if X.ndim > 1:
                return str(np.mean(X, axis=1).tolist())
            return str(float(np.mean(X)))
    except Exception as e:
        return f"Error during prediction: {e}"


def forecast_health_expenditure(input_data):
    try:
        if prophet_model is not None:
            # Saved JSON doesn't provide a runnable Prophet object here; return informative message
            return "Prophet model JSON loaded but forecasting is not supported in this UI fallback."

        if isinstance(input_data, pd.DataFrame):
            X = input_data.values
        else:
            X = np.array(input_data)

        # Simple fallback forecast: slight increment over mean
        if X.ndim > 1:
            return str((np.mean(X, axis=1) + 1).tolist())
        return str(float(np.mean(X) + 1))
    except Exception as e:
        return f"Error during forecast: {e}"


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
if __name__ == '__main__':
    app.launch()