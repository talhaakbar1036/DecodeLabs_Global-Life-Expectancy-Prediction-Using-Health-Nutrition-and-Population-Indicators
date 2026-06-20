"""
Simple Gradio app — Infant Mortality Rate Predictor
Uses the Random Forest model trained in the project notebook.

Run:
    pip install -r requirements.txt
    python app.py
Then open http://localhost:7860
"""

import os
import joblib
import pandas as pd
import gradio as gr
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(BASE_DIR, "data", "de46020d-9719-41d6-a517-0a7fb19791b1_Data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "data", "random_forest_regressor_infant_mortality.pkl")

FEATURES = [
    "current_health_expenditure_pct_gdp",
    "physicians_per_1k_people",
    "hospital_beds_per_1k_people",
    "people_using_at_least_basic_drinking_water_services_pct_pop",
    "diabetes_prevalence_(%_of_population_ages_20_to_79)",
    "incidence_of_tuberculosis_per_100k_people",
]

LABELS = {
    "current_health_expenditure_pct_gdp": "Health expenditure (% of GDP)",
    "physicians_per_1k_people": "Physicians (per 1,000 people)",
    "hospital_beds_per_1k_people": "Hospital beds (per 1,000 people)",
    "people_using_at_least_basic_drinking_water_services_pct_pop": "Basic drinking water access (% of population)",
    "diabetes_prevalence_(%_of_population_ages_20_to_79)": "Diabetes prevalence (% ages 20-79)",
    "incidence_of_tuberculosis_per_100k_people": "Tuberculosis incidence (per 100,000 people)",
}

TARGET = "mortality_rate_infant_per_1k_live_births"


# ---------------- Load data and rebuild the same features used in training ----------------

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
    pivot.columns = [c.replace("[", "").replace("]", "").replace(",", "").replace("-", "_").replace(" ", "_").lower() for c in cols]
    return pivot


print("Loading data and model...")
df_pivot = load_data()
model = joblib.load(MODEL_PATH)

# Fit a scaler matching the one used at training time (same data, same columns)
diabetes_col = "diabetes_prevalence_(%_of_population_ages_20_to_79)"
df_clean = df_pivot.copy()
df_clean[diabetes_col] = df_clean[diabetes_col].fillna(df_clean[diabetes_col].mean())
df_clean = df_clean.dropna(subset=[TARGET] + [f for f in FEATURES if f != diabetes_col])
scaler = StandardScaler().fit(df_clean[FEATURES])

countries = sorted(df_pivot["country_name"].dropna().unique().tolist())
ranges = {f: (float(df_pivot[f].min()), float(df_pivot[f].max()), float(df_clean[f].mean())) for f in FEATURES}
ranges[diabetes_col] = (
    float(df_pivot[diabetes_col].dropna().min()),
    float(df_pivot[diabetes_col].dropna().max()),
    float(df_pivot[diabetes_col].dropna().mean()),
)


# ---------------- App logic ----------------

def autofill(country):
    rows = df_pivot[df_pivot["country_name"] == country].sort_values("year", ascending=False)
    if not len(rows):
        return [gr.update() for _ in FEATURES]
    values = []
    for f in FEATURES:
        series = rows[["year", f]].dropna()
        v = float(series.iloc[0][f]) if len(series) else ranges[f][2]
        lo, hi, _ = ranges[f]
        values.append(round(min(max(v, lo), hi), 2))
    return values


def predict(*vals):
    X = pd.DataFrame([dict(zip(FEATURES, vals))])[FEATURES]
    X_scaled = scaler.transform(X)
    result = model.predict(X_scaled)[0]
    return f"### {result:.1f} deaths per 1,000 live births"


# ---------------- UI ----------------

with gr.Blocks(title="Infant Mortality Predictor") as app:
    gr.Markdown("# 🌍 Infant Mortality Predictor")
    gr.Markdown("Predicts infant mortality rate from health indicators using a Random Forest model.")

    country = gr.Dropdown(choices=countries, value="Pakistan", label="Pick a country to autofill")

    sliders = []
    for f in FEATURES:
        lo, hi, mean = ranges[f]
        sliders.append(gr.Slider(minimum=round(lo, 2), maximum=round(hi, 2), value=round(mean, 2), label=LABELS[f]))

    btn = gr.Button("Predict", variant="primary")
    output = gr.Markdown()

    country.change(autofill, inputs=country, outputs=sliders)
    btn.click(predict, inputs=sliders, outputs=output)
    app.load(autofill, inputs=country, outputs=sliders).then(predict, inputs=sliders, outputs=output)

if __name__ == "__main__":
    app.launch()
