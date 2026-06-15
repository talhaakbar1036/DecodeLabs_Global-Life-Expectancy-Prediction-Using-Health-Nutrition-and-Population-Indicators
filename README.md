# Global Life Expectancy Prediction

A professional data science repository for predicting life expectancy using global health, nutrition, and population indicators from 2015 to 2024. This project combines data preparation, exploratory analysis, machine learning modeling, and an interactive Gradio web application to deliver actionable forecasting insights.

## Project Overview

This repository is designed to demonstrate a complete analytical workflow:

- Data ingestion and cleaning for global health indicators
- Exploratory data analysis (EDA) to uncover trends and correlations
- Feature engineering for machine learning readiness
- Supervised learning and time series forecasting for health expenditure and related outcomes
- A deployable Gradio app for interactive prediction and visual exploration

The project includes a focused case study for Pakistan, using historical health expenditure data to support forecasting.

## Online Resources

The core data and model artifacts are available directly in the GitHub repository.

### Dataset files

- `dataset/de46020d-9719-41d6-a517-0a7fb19791b1_Data (1).csv`
  - https://github.com/talhaakbar1036/Global-Life-Expectancy-Prediction-Using-Health-Nutrition-and-Population-Indicators-2015-2024-/blob/main/dataset/de46020d-9719-41d6-a517-0a7fb19791b1_Data%20(1).csv
- `dataset/de46020d-9719-41d6-a517-0a7fb19791b1_Data.csv`
  - https://github.com/talhaakbar1036/Global-Life-Expectancy-Prediction-Using-Health-Nutrition-and-Population-Indicators-2015-2024-/blob/main/dataset/de46020d-9719-41d6-a517-0a7fb19791b1_Data.csv

### Saved machine learning models

- `saved_ml_models/arima_pakistan_health_expenditure.pkl`
  - https://github.com/talhaakbar1036/Global-Life-Expectancy-Prediction-Using-Health-Nutrition-and-Population-Indicators-2015-2024-/blob/main/saved_ml_models/arima_pakistan_health_expenditure.pkl
- `saved_ml_models/lstm_pakistan_health_expenditure.h5`
  - https://github.com/talhaakbar1036/Global-Life-Expectancy-Prediction-Using-Health-Nutrition-and-Population-Indicators-2015-2024-/blob/main/saved_ml_models/lstm_pakistan_health_expenditure.h5
- `saved_ml_models/prophet_pakistan_health_expenditure.json`
  - https://github.com/talhaakbar1036/Global-Life-Expectancy-Prediction-Using-Health-Nutrition-and-Population-Indicators-2015-2024-/blob/main/saved_ml_models/prophet_pakistan_health_expenditure.json
- `saved_ml_models/random_forest_classifier_health_expenditure.pkl`
  - https://github.com/talhaakbar1036/Global-Life-Expectancy-Prediction-Using-Health-Nutrition-and-Population-Indicators-2015-2024-/blob/main/saved_ml_models/random_forest_classifier_health_expenditure.pkl
- `saved_ml_models/random_forest_regressor_infant_mortality.pkl`
  - https://github.com/talhaakbar1036/Global-Life-Expectancy-Prediction-Using-Health-Nutrition-and-Population-Indicators-2015-2024-/blob/main/saved_ml_models/random_forest_regressor_infant_mortality.pkl

### Prediction output files

- `ml_prediction/health_expenditure_classification_predictions.csv`
  - https://github.com/talhaakbar1036/Global-Life-Expectancy-Prediction-Using-Health-Nutrition-and-Population-Indicators-2015-2024-/blob/main/ml_prediction/health_expenditure_classification_predictions.csv
- `ml_prediction/infant_mortality_predictions.csv`
  - https://github.com/talhaakbar1036/Global-Life-Expectancy-Prediction-Using-Health-Nutrition-and-Population-Indicators-2015-2024-/blob/main/ml_prediction/infant_mortality_predictions.csv
- `ml_prediction/pakistan_health_expenditure_forecasts.csv`
  - https://github.com/talhaakbar1036/Global-Life-Expectancy-Prediction-Using-Health-Nutrition-and-Population-Indicators-2015-2024-/blob/main/ml_prediction/pakistan_health_expenditure_forecasts.csv

### Gradio app and UI resources

The interactive app is implemented in `global-life-expectancy-gradio-app/` and can be accessed directly on GitHub.

- App directory: https://github.com/talhaakbar1036/Global-Life-Expectancy-Prediction-Using-Health-Nutrition-and-Population-Indicators-2015-2024-/tree/main/global-life-expectancy-gradio-app
- `global-life-expectancy-gradio-app/requirements.txt`
  - https://github.com/talhaakbar1036/Global-Life-Expectancy-Prediction-Using-Health-Nutrition-and-Population-Indicators-2015-2024-/blob/main/global-life-expectancy-gradio-app/requirements.txt
- `global-life-expectancy-gradio-app/app.py`
  - https://github.com/talhaakbar1036/Global-Life-Expectancy-Prediction-Using-Health-Nutrition-and-Population-Indicators-2015-2024-/blob/main/global-life-expectancy-gradio-app/app.py
- `global-life-expectancy-gradio-app/README.md`
  - https://github.com/talhaakbar1036/Global-Life-Expectancy-Prediction-Using-Health-Nutrition-and-Population-Indicators-2015-2024-/blob/main/global-life-expectancy-gradio-app/README.md
- `global-life-expectancy-gradio-app/assets/animations.js`
  - https://github.com/talhaakbar1036/Global-Life-Expectancy-Prediction-Using-Health-Nutrition-and-Population-Indicators-2015-2024-/blob/main/global-life-expectancy-gradio-app/assets/animations.js
- `global-life-expectancy-gradio-app/assets/styles.css`
  - https://github.com/talhaakbar1036/Global-Life-Expectancy-Prediction-Using-Health-Nutrition-and-Population-Indicators-2015-2024-/blob/main/global-life-expectancy-gradio-app/assets/styles.css
- App data and models:
  - https://github.com/talhaakbar1036/Global-Life-Expectancy-Prediction-Using-Health-Nutrition-and-Population-Indicators-2015-2024-/blob/main/global-life-expectancy-gradio-app/data/dataset/de46020d-9719-41d6-a517-0a7fb19791b1_Data.csv
  - https://github.com/talhaakbar1036/Global-Life-Expectancy-Prediction-Using-Health-Nutrition-and-Population-Indicators-2015-2024-/blob/main/global-life-expectancy-gradio-app/data/ml_prediction/infant_mortality_predictions.csv
  - https://github.com/talhaakbar1036/Global-Life-Expectancy-Prediction-Using-Health-Nutrition-and-Population-Indicators-2015-2024-/blob/main/global-life-expectancy-gradio-app/data/ml_prediction/pakistan_health_expenditure_forecasts.csv
  - https://github.com/talhaakbar1036/Global-Life-Expectancy-Prediction-Using-Health-Nutrition-and-Population-Indicators-2015-2024-/blob/main/global-life-expectancy-gradio-app/data/saved_ml_models/prophet_pakistan_health_expenditure.json
  - https://github.com/talhaakbar1036/Global-Life-Expectancy-Prediction-Using-Health-Nutrition-and-Population-Indicators-2015-2024-/blob/main/global-life-expectancy-gradio-app/data/saved_ml_models/lstm_pakistan_health_expenditure.h5

## Dataset Description

The dataset contains country-level health and socio-economic indicators such as:

- Health expenditure per capita
- Infant mortality rate
- Nutrition and food security metrics
- Population totals and demographics
- Additional public health indicators used for modeling

These features support both classification and regression tasks for forecasting key outcomes.

## Data Source and Forecast Fix

This project uses World Bank data for East countries covering the years 2015 through 2024. A duplicate dataset file was removed to save space and avoid confusion.

The Gradio app now supports health expenditure forecasting for the next years 2025 and 2026 using the precomputed forecast CSV. If you enter a future year in the app, it will return the forecast value for that year when available. Forecasting beyond 2026 is not supported until the forecast dataset is extended.

## Machine Learning Components

Included workflows:

- Time series forecasting for Pakistan health expenditure using ARIMA, LSTM, and Prophet
- Classification of health expenditure categories using Random Forest
- Regression modeling of infant mortality using Random Forest

Saved model files are available under `saved_ml_models/` and can be reused with the app or additional analysis scripts.

## How to Run

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Global-Life-Expectancy-Prediction-Using-Health-Nutrition-and-Population-Indicators-2015-2024-
   ```

2. Install the app dependencies:
   ```bash
   pip install -r global-life-expectancy-gradio-app/requirements.txt
   ```

3. Start the Gradio app:
   ```bash
   cd global-life-expectancy-gradio-app
   python app.py
   ```

4. Open `http://localhost:7860` in a browser.

## Repository Structure

- `dataset/` - main dataset CSV files used for training and analysis
- `ml_prediction/` - model prediction and forecast output CSV files
- `saved_ml_models/` - stored trained models and serialized artifacts
- `global-life-expectancy-gradio-app/` - Gradio application, UI assets, and app-specific data
- `README.md` - this documentation file
- `LICENSE` - project license

## License

This project is released under the MIT License. See `LICENSE` for details.
