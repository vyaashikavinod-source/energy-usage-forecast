# Energy Usage Forecast

Daily electricity consumption varies with temperature, short-term demand shifts, and behavioral patterns. These variations are not strictly linear. This project evaluates whether a non-linear machine learning approach can better capture short-term electricity usage dynamics compared to a traditional linear baseline.

## Overview

The dataset includes:
- Daily electricity usage (kWh)
- Daily mean temperature
- Calendar-derived time features

Additional engineered features were constructed to capture temporal structure:
- Lag features
- Rolling averages
- Short-term trend signals

Two modeling approaches were implemented:
1. Linear baseline regression
2. Gradient-based machine learning model

The baseline assumes a linear relationship between predictors and consumption.  
The machine learning model captures non-linear interactions and short-term fluctuations.

## Model Evaluation

Model performance was evaluated using validation mean absolute error (MAE).

- Validation MAE (AI model): **0.312**
- Validation MAE (Baseline): **0.441**
- Relative improvement: **29%**

The reduction in MAE indicates that the non-linear model generalizes better on unseen data and captures additional structure in the data.

## Application Features

The deployed Streamlit application allows users to:
- Adjust forecast horizon
- Modify assumed temperature
- Compare baseline and AI forecasts
- Estimate total electricity cost
- Export forecast results as CSV

The system is structured to separate modeling logic from the application layer, allowing experimentation and deployment within the same project.

## Project Structure

```text
energy-usage-forecast/
├─ .github/                 # GitHub workflows / templates (CI, etc.)
├─ app/                     # Streamlit application (UI layer)
├─ artifacts/               # saved models + preprocessors (pickle/joblib, etc.)
├─ dataset/                 # raw/processed datasets
├─ docs/                    # extra documentation
├─ EDA/                     # notebooks / exploratory analysis
├─ images/                  # images used in README/docs
├─ modeling/                # training + evaluation code (baselines, CV, etc.)
├─ reports/                 # generated reports/exports/results
├─ scripts/                 # runnable scripts (train/evaluate/run app helpers)
├─ src/                     # shared utilities (data, features, forecasting)
├─ tests/                   # unit/integration tests
└─ tools/                   # helper tooling (one-off utilities)
