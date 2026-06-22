# Global Life Expectancy Prediction

Predicting infant mortality and health expenditure patterns from World Bank health, nutrition, and population indicators (2015–2024), with a Pakistan-specific health expenditure forecast.

## Overview

This project works through three modeling tasks on the same World Bank dataset:

1. **Infant mortality regression** — predicting a country's infant mortality rate (per 1,000 live births) from health system and nutrition indicators.
2. **Health expenditure classification** — predicting whether a country spends above or below the global median on health (% of GDP) based on demographic and disease-burden indicators.
3. **Pakistan health expenditure forecasting** — projecting Pakistan's health expenditure (% of GDP) through 2026 using ARIMA, Prophet, and LSTM.

The full workflow — data cleaning, exploratory analysis, model comparison, hyperparameter tuning, and forecasting — is in the notebook. A small Gradio app exposes the infant mortality model as an interactive demo.

## Results

**Infant mortality regression** (test set, after hyperparameter tuning)

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression | 8.48 | 5.81 | 0.73 |
| Neural Network | 8.14 | 5.32 | 0.75 |
| XGBoost | 5.91 | 3.08 | 0.87 |
| **Random Forest (tuned)** | **5.47** | **3.14** | **0.89** |

Random Forest was selected as the final model and saved for reuse.

**Health expenditure classification** (test set)

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | 0.60 | 0.67 |
| Random Forest | 0.60 | 0.50 |
| SVM | 0.40 | 0.33 |

Classification performance is modest across all models — the dataset is small after filtering for complete feature rows, and the features used (population growth, urbanization, infant mortality, tuberculosis incidence, diabetes prevalence) only partially explain health spending levels. This is noted here rather than smoothed over, since it's a real limitation of the current feature set.

**Pakistan health expenditure forecast (% of GDP)**

| Year | ARIMA | Prophet | LSTM |
|---|---|---|---|
| 2024 | 2.18 | 2.93 | 2.68 |
| 2025 | 2.10 | 2.95 | 2.75 |
| 2026 | 2.08 | — | 2.78 |

The three models diverge meaningfully on direction (ARIMA trends down, Prophet and LSTM trend flat to slightly up), which is worth keeping in mind when reading the forecast — with only nine years of annual data, none of these models has much to work with.

## Repository structure

```
.
├── Global_Life_Expectancy_Prediction_Using_Health_Nutrition_and_Population_Indicators_2015_2024.ipynb
├── dataset/
│   └── de46020d-9719-41d6-a517-0a7fb19791b1_Data.csv      World Bank source data
├── saved_ml_models/
│   ├── random_forest_regressor_infant_mortality.pkl
│   ├── random_forest_classifier_health_expenditure.pkl
│   ├── arima_pakistan_health_expenditure.pkl
│   ├── prophet_pakistan_health_expenditure.json
│   └── lstm_pakistan_health_expenditure.h5
├── ml_prediction/
│   ├── infant_mortality_predictions.csv
│   ├── health_expenditure_classification_predictions.csv
│   └── pakistan_health_expenditure_forecasts.csv
├── app/
│   ├── app.py            Gradio demo for the infant mortality model
│   ├── requirements.txt
│   └── data/              copies of the dataset and model needed by the app
└── LICENSE
```

## Dataset

World Bank Health, Nutrition and Population Statistics, 2015–2024, covering indicators such as:

- Current health expenditure (% of GDP)
- Physicians and hospital beds per 1,000 people
- Infant mortality rate (per 1,000 live births)
- Tuberculosis incidence and diabetes prevalence
- Access to basic drinking water
- Population growth and urbanization rates

The raw file is in World Bank's wide format (one row per indicator per country, one column per year) and is reshaped into a country-year table during preprocessing. Indicators are reported on different schedules across countries, so the notebook handles missing values per task rather than dropping any country outright.

## Running the app

```bash
cd app
pip install -r requirements.txt
python app.py
```

Open `http://localhost:7860`. Pick a country to auto-fill the sliders with its most recently reported values, adjust as needed, and predict.

## Running the notebook

The notebook expects the dataset CSV in `dataset/`. Model artifacts are written to `saved_ml_models/` and predictions to `ml_prediction/`. Cells were originally run on Google Colab with paths pointed at Google Drive; update the path variables near the top of the data-loading and model-saving cells if running locally.

## License

MIT — see [LICENSE](LICENSE).
