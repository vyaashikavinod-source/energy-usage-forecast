# energy-forecast-app
README VERSION 1 — Technical / Research Focus
Energy Usage Forecast

Daily electricity consumption varies with temperature, short-term demand shifts, and behavioral patterns. These variations are not strictly linear. This project investigates how well different modeling approaches can capture short-term electricity usage dynamics.

The dataset includes:

Daily electricity usage (kWh)

Daily mean temperature

Calendar-derived time features

Additional engineered features were constructed, including lag values and rolling statistics, to capture temporal structure in the data.

Two modeling approaches were implemented:

Linear baseline regression

Gradient-based machine learning model

The baseline assumes a primarily linear relationship between predictors and consumption. The machine learning model captures non-linear interactions and short-term fluctuations that are not well explained by linear assumptions.

Model performance was evaluated using validation mean absolute error (MAE).

Example validation results:

Validation MAE (AI model): 0.312
Validation MAE (Baseline): 0.441
Relative improvement: 29 percent

The reduction in MAE indicates that the non-linear model generalizes better on unseen data.

The project includes:

Feature engineering pipeline

Model training and evaluation

Baseline comparison

Deployment-ready inference workflow

Interactive Streamlit interface

The application allows users to adjust forecast horizon, modify assumed temperature, compare baseline and AI forecasts, estimate total cost, and export results.

This project demonstrates applied time-series modeling, quantitative evaluation, and practical deployment in a reproducible workflow.

Project Structure

app
modeling
dataset
artifacts
src
tests
docs

Running the Application

Install dependencies:

pip install -r requirements.txt

Run the app:

streamlit run app/app.py
