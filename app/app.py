"""
Global Life Expectancy Prediction — Gradio app.

Three demos backed by the models trained in the project notebook:
  1. Infant mortality rate prediction (Random Forest Regressor)
  2. Health expenditure classification — above/below global median (Random Forest Classifier)
  3. Pakistan health expenditure forecasting (ARIMA, Prophet, LSTM)

The forecasting models (ARIMA / Prophet / LSTM) are loaded lazily — only the
first time a particular model is actually used — so the app starts quickly
and never imports tensorflow/prophet/statsmodels unless that tab is used.

Run:
    pip install -r requirements.txt
    python app.py
Then open http://localhost:7860
"""

import os
import joblib
import numpy as np
import pandas as pd
import gradio as gr
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_CSV = os.path.join(DATA_DIR, "de46020d-9719-41d6-a517-0a7fb19791b1_Data.csv")
FORECAST_CSV = os.path.join(DATA_DIR, "pakistan_health_expenditure_forecasts.csv")

RF_REGRESSOR_PATH = os.path.join(DATA_DIR, "random_forest_regressor_infant_mortality.pkl")
RF_CLASSIFIER_PATH = os.path.join(DATA_DIR, "random_forest_classifier_health_expenditure.pkl")
ARIMA_PATH = os.path.join(DATA_DIR, "arima_pakistan_health_expenditure.pkl")
PROPHET_PATH = os.path.join(DATA_DIR, "prophet_pakistan_health_expenditure.json")
LSTM_PATH = os.path.join(DATA_DIR, "lstm_pakistan_health_expenditure.h5")

# ----------------------------------------------------------------------------
# Task A — Infant mortality regression
# ----------------------------------------------------------------------------

FEATURES_A = [
    "current_health_expenditure_pct_gdp",
    "physicians_per_1k_people",
    "hospital_beds_per_1k_people",
    "people_using_at_least_basic_drinking_water_services_pct_pop",
    "diabetes_prevalence_(%_of_population_ages_20_to_79)",
    "incidence_of_tuberculosis_per_100k_people",
]

LABELS_A = {
    "current_health_expenditure_pct_gdp": "Health expenditure (% of GDP)",
    "physicians_per_1k_people": "Physicians (per 1,000 people)",
    "hospital_beds_per_1k_people": "Hospital beds (per 1,000 people)",
    "people_using_at_least_basic_drinking_water_services_pct_pop": "Basic drinking water access (% of population)",
    "diabetes_prevalence_(%_of_population_ages_20_to_79)": "Diabetes prevalence (% ages 20-79)",
    "incidence_of_tuberculosis_per_100k_people": "Tuberculosis incidence (per 100,000 people)",
}

TARGET_A = "mortality_rate_infant_per_1k_live_births"

# ----------------------------------------------------------------------------
# Task B — Health expenditure classification
# ----------------------------------------------------------------------------

FEATURES_B = [
    "population_growth_annual_pct",
    "urban_population_pct_total_pop",
    "mortality_rate_infant_per_1k_live_births",
    "incidence_of_tuberculosis_per_100k_people",
    "diabetes_prevalence_(%_of_population_ages_20_to_79)",
]

LABELS_B = {
    "population_growth_annual_pct": "Population growth (annual %)",
    "urban_population_pct_total_pop": "Urban population (% of total)",
    "mortality_rate_infant_per_1k_live_births": "Infant mortality (per 1,000 live births)",
    "incidence_of_tuberculosis_per_100k_people": "Tuberculosis incidence (per 100,000 people)",
    "diabetes_prevalence_(%_of_population_ages_20_to_79)": "Diabetes prevalence (% ages 20-79)",
}

TARGET_B = "current_health_expenditure_pct_gdp"


# ----------------------------------------------------------------------------
# Data pipeline — rebuilds the same country-year feature table used in
# training, straight from the raw World Bank CSV.
# ----------------------------------------------------------------------------


def load_data():
    df = pd.read_csv(DATA_CSV, encoding="latin1")
    year_cols = [c for c in df.columns if "[" in c and "]" in c]
    for col in year_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Series Name", "Series Code"])

    id_vars = ["Series Name", "Series Code", "Country Name", "Country Code"]
    long_df = pd.melt(df, id_vars=id_vars, value_vars=year_cols, var_name="Year_Raw", value_name="Value")
    long_df["Year"] = long_df["Year_Raw"].str.extract(r"(\d{4})").astype(int)
    long_df = long_df[~long_df["Series Name"].str.contains("Data from database|Last Updated", na=False)]

    pivot = long_df.pivot_table(index=["Country Name", "Year"], columns="Series Name", values="Value").reset_index()
    pivot.columns.name = None

    cols = []
    for col in pivot.columns:
        c = (
            col.replace(" (% of GDP)", "_pct_GDP")
            .replace(" (per 1,000 people)", "_per_1k_people")
            .replace(" (per 100,000 people)", "_per_100k_people")
            .replace(" (% of adults ages 30-79)", "_pct_adults")
            .replace(" (% of population)", "_pct_pop")
            .replace(" (per 1,000 live births)", "_per_1k_live_births")
            .replace(" (modeled estimate, per 100,000 live births)", "_per_100k_live_births")
            .replace(" (current US$)", "_current_USD")
            .replace(" (annual %)", "_annual_pct")
            .replace(" (% of total population)", "_pct_total_pop")
        )
        cols.append(c)
    pivot.columns = [
        c.replace("[", "").replace("]", "").replace(",", "").replace("-", "_").replace(" ", "_").lower() for c in cols
    ]
    return pivot


print("Loading data and core models...")
df_pivot = load_data()
rf_regressor = joblib.load(RF_REGRESSOR_PATH)
rf_classifier = joblib.load(RF_CLASSIFIER_PATH)

countries = sorted(df_pivot["country_name"].dropna().unique().tolist())

# --- Scaler A (infant mortality) ---
diabetes_col = "diabetes_prevalence_(%_of_population_ages_20_to_79)"
real_diabetes = df_pivot[diabetes_col].dropna()

df_a = df_pivot.copy()
df_a[diabetes_col] = df_a[diabetes_col].fillna(df_a[diabetes_col].mean())
df_a = df_a.dropna(subset=[TARGET_A] + [f for f in FEATURES_A if f != diabetes_col])
scaler_a = StandardScaler().fit(df_a[FEATURES_A])

ranges_a = {f: (float(df_pivot[f].min()), float(df_pivot[f].max()), float(df_a[f].mean())) for f in FEATURES_A}
ranges_a[diabetes_col] = (float(real_diabetes.min()), float(real_diabetes.max()), float(real_diabetes.mean()))

# --- Scaler B (health expenditure classification) ---
df_b = df_pivot.dropna(subset=[TARGET_B] + FEATURES_B).copy()
median_expenditure = float(df_b[TARGET_B].median())
scaler_b = StandardScaler().fit(df_b[FEATURES_B])
ranges_b = {f: (float(df_pivot[f].min()), float(df_pivot[f].max()), float(df_b[f].mean())) for f in FEATURES_B}

# --- Pakistan history + precomputed forecast comparison table ---
pakistan_history = (
    df_pivot[df_pivot["country_name"] == "Pakistan"][["year", "current_health_expenditure_pct_gdp"]]
    .dropna()
    .sort_values("year")
)
last_historical_year = int(pakistan_history["year"].max())
precomputed_forecast = pd.read_csv(FORECAST_CSV, index_col=0)


def autofill(features, ranges, country):
    rows = df_pivot[df_pivot["country_name"] == country].sort_values("year", ascending=False)
    if not len(rows):
        return [gr.update() for _ in features]
    values = []
    for f in features:
        series = rows[["year", f]].dropna()
        v = float(series.iloc[0][f]) if len(series) else ranges[f][2]
        lo, hi, _ = ranges[f]
        values.append(round(min(max(v, lo), hi), 2))
    return values


# ----------------------------------------------------------------------------
# Lazy-loaded forecasting models — only imported/loaded on first use, so the
# app starts instantly and skips tensorflow/prophet/statsmodels unless the
# Pakistan Forecast tab is actually used.
# ----------------------------------------------------------------------------

_lstm_model = None
_prophet_model = None
_arima_model = None


def get_lstm():
    global _lstm_model
    if _lstm_model is None:
        import tensorflow as tf

        _lstm_model = tf.keras.models.load_model(LSTM_PATH, compile=False)
    return _lstm_model


def get_prophet():
    global _prophet_model
    if _prophet_model is None:
        from prophet.serialize import model_from_json

        with open(PROPHET_PATH) as f:
            _prophet_model = model_from_json(f.read())
    return _prophet_model


def get_arima():
    global _arima_model
    if _arima_model is None:
        from statsmodels.tsa.arima.model import ARIMAResults

        _arima_model = ARIMAResults.load(ARIMA_PATH)
    return _arima_model


# ----------------------------------------------------------------------------
# Task A — Infant mortality prediction
# ----------------------------------------------------------------------------


def autofill_a(country):
    return autofill(FEATURES_A, ranges_a, country)


def predict_infant_mortality(*vals):
    X = pd.DataFrame([dict(zip(FEATURES_A, vals))])[FEATURES_A]
    X_scaled = scaler_a.transform(X)
    result = rf_regressor.predict(X_scaled)[0]
    return f"### {result:.1f} deaths per 1,000 live births"


# ----------------------------------------------------------------------------
# Task B — Health expenditure classification
# ----------------------------------------------------------------------------


def autofill_b(country):
    return autofill(FEATURES_B, ranges_b, country)


def predict_expenditure_class(*vals):
    X = pd.DataFrame([dict(zip(FEATURES_B, vals))])[FEATURES_B]
    X_scaled = scaler_b.transform(X)
    pred_class = int(rf_classifier.predict(X_scaled)[0])
    prob_high = float(rf_classifier.predict_proba(X_scaled)[0, 1])

    label = "Above-median health spending" if pred_class == 1 else "Below-median health spending"
    sub = (
        f"Model confidence the country spends **above** the global median "
        f"({median_expenditure:.2f}% of GDP): **{prob_high * 100:.1f}%**"
    )
    return f"### {label}", sub


# ----------------------------------------------------------------------------
# Task C — Pakistan health expenditure forecast
# ----------------------------------------------------------------------------

YEAR_CHOICES = [str(y) for y in range(int(pakistan_history["year"].min()), 2027)]


def forecast_pakistan(model_choice, year_choice):
    target_year = int(year_choice)

    if target_year <= last_historical_year:
        row = pakistan_history[pakistan_history["year"] == target_year]
        if len(row):
            value = float(row.iloc[0]["current_health_expenditure_pct_gdp"])
            note = f"{target_year} is a recorded value, not a forecast."
            return f"### {value:.2f}% of GDP", note

        note = f"No recorded value for {target_year} in the source data."
        return "### Unavailable", note

    steps = target_year - last_historical_year
    if steps > 3:
        return "### Unavailable", "Forecasting is only supported up to 2026 (3 years past the last recorded year, 2023)."

    if model_choice == "ARIMA":
        value = float(get_arima().forecast(steps=steps).iloc[-1])
        note = f"ARIMA(1,1,1) forecast for {target_year}, based on Pakistan's 2015–2023 series."
    elif model_choice == "Prophet":
        m = get_prophet()
        future = m.make_future_dataframe(periods=steps, freq="YE")
        fc = m.predict(future)
        value = float(fc.iloc[-1]["yhat"])
        note = f"Prophet forecast for {target_year}, based on Pakistan's 2015–2023 series."
    else:  # LSTM
        from sklearn.preprocessing import MinMaxScaler

        data = pakistan_history["current_health_expenditure_pct_gdp"].values.reshape(-1, 1)
        mm_scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = mm_scaler.fit_transform(data)
        lstm = get_lstm()
        current_input = data_scaled[-1:].reshape(1, 1, 1)
        preds_scaled = []
        for _ in range(steps):
            pred = lstm.predict(current_input, verbose=0)[0, 0]
            preds_scaled.append(pred)
            current_input = np.array(pred).reshape(1, 1, 1)
        value = float(mm_scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()[-1])
        note = f"LSTM forecast for {target_year}, based on Pakistan's 2015–2023 series."

    return f"### {value:.2f}% of GDP", note


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------

with gr.Blocks(title="Global Life Expectancy Prediction") as app:
    gr.Markdown("# 🌍 Global Life Expectancy Prediction")
    gr.Markdown(
        "Three models trained on World Bank health, nutrition, and population indicators (2015–2024)."
    )

    with gr.Tabs():
        # ---------------- Tab 1: Infant mortality ----------------
        with gr.Tab("Infant Mortality Predictor"):
            gr.Markdown(
                "Predicts a country's infant mortality rate (per 1,000 live births) from health "
                "system and nutrition indicators, using a tuned **Random Forest Regressor** "
                "(test R² ≈ 0.89)."
            )
            country_a = gr.Dropdown(choices=countries, value="Pakistan", label="Pick a country to autofill")

            sliders_a = []
            for f in FEATURES_A:
                lo, hi, mean = ranges_a[f]
                sliders_a.append(
                    gr.Slider(minimum=round(lo, 2), maximum=round(hi, 2), value=round(mean, 2), label=LABELS_A[f])
                )

            btn_a = gr.Button("Predict", variant="primary")
            output_a = gr.Markdown()

            country_a.change(autofill_a, inputs=country_a, outputs=sliders_a)
            btn_a.click(predict_infant_mortality, inputs=sliders_a, outputs=output_a)
            app.load(autofill_a, inputs=country_a, outputs=sliders_a).then(
                predict_infant_mortality, inputs=sliders_a, outputs=output_a
            )

        # ---------------- Tab 2: Health expenditure classification ----------------
        with gr.Tab("Health Expenditure Classifier"):
            gr.Markdown(
                "Classifies whether a country's health expenditure (% of GDP) is likely above or "
                f"below the global median (**{median_expenditure:.2f}%**), using a **Random Forest "
                "Classifier** trained on demographic and disease-burden indicators."
            )
            country_b = gr.Dropdown(choices=countries, value="Pakistan", label="Pick a country to autofill")

            sliders_b = []
            for f in FEATURES_B:
                lo, hi, mean = ranges_b[f]
                sliders_b.append(
                    gr.Slider(minimum=round(lo, 2), maximum=round(hi, 2), value=round(mean, 2), label=LABELS_B[f])
                )

            btn_b = gr.Button("Classify", variant="primary")
            output_b = gr.Markdown()
            output_b_sub = gr.Markdown()

            country_b.change(autofill_b, inputs=country_b, outputs=sliders_b)
            btn_b.click(predict_expenditure_class, inputs=sliders_b, outputs=[output_b, output_b_sub])
            app.load(autofill_b, inputs=country_b, outputs=sliders_b).then(
                predict_expenditure_class, inputs=sliders_b, outputs=[output_b, output_b_sub]
            )

        # ---------------- Tab 3: Pakistan forecast ----------------
        with gr.Tab("Pakistan Forecast"):
            gr.Markdown(
                "Forecasts Pakistan's current health expenditure (% of GDP) using **ARIMA**, "
                "**Prophet**, or **LSTM** — each trained on its 2015–2023 annual series.\n\n"
                "**How to use this tab:**\n"
                "- Pick a year between 2015 and 2026.\n"
                "- Years up to **2023** return the actual recorded value (2024's figure was not "
                "yet reported when the dataset was collected, so it's treated as a forecast target too).\n"
                "- Years **2024–2026** return that model's forecast.\n"
                "- Forecasting beyond 2026 isn't supported — the models were only trained on nine "
                "years of annual data, so projections get unreliable quickly. Treat any forecast "
                "here as a rough trend indicator, not a precise estimate.\n"
                "- The three models can disagree noticeably; switching between them for the same "
                "year is a quick way to see how much that uncertainty matters."
            )
            with gr.Row():
                model_choice = gr.Radio(choices=["ARIMA", "Prophet", "LSTM"], value="Prophet", label="Forecasting model")
                year_choice = gr.Dropdown(choices=YEAR_CHOICES, value="2025", label="Year")

            btn_c = gr.Button("Get value", variant="primary")
            output_c = gr.Markdown()
            output_c_sub = gr.Markdown()

            btn_c.click(forecast_pakistan, inputs=[model_choice, year_choice], outputs=[output_c, output_c_sub])
            app.load(forecast_pakistan, inputs=[model_choice, year_choice], outputs=[output_c, output_c_sub])

if __name__ == "__main__":
    app.launch()
