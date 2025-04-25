from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objs as go
from sqlalchemy import create_engine
from datetime import datetime

# Load Pre-trained Forecasting Model
model = pickle.load(open('best_model.pkl', 'rb'))

# Create Flask App
app = Flask(__name__, static_folder='static')

# Database Configuration
db_host = "localhost"
db_user = "root"
db_password = "anushasj02"
db_name = "aluminium_price"
db_table = "df"

# Establish Database Connection
def get_db_connection():
    engine = create_engine(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")
    return engine

# Fetch Data from Database
def fetch_data_from_db():
    engine = get_db_connection()
    query = f"SELECT Date, Price FROM {db_table} ORDER BY Date"
    df = pd.read_sql(query, con=engine)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

# Generate Forecast with Confidence Intervals
def generate_forecast(start_date, end_date, confidence=0.95):
    future_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    forecast_values = model.forecast(steps=len(future_dates))

    ci_multiplier = 1.645 if confidence == 0.90 else 1.96
    forecast_std = np.std(forecast_values) * 0.1
    lower_bound = forecast_values - (ci_multiplier * forecast_std)
    upper_bound = forecast_values + (ci_multiplier * forecast_std)

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted": forecast_values,
        "Lower_Bound": lower_bound,
        "Upper_Bound": upper_bound
    })

    return forecast_df

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/forecast', methods=['POST'])
def forecast():
    start_date = pd.to_datetime(request.form['start_date'])
    end_date = pd.to_datetime(request.form['end_date'])
    confidence_level = float(request.form['confidence_level'])

    forecast_df = generate_forecast(start_date, end_date, confidence=confidence_level)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Predicted'], mode='lines', name='Forecast', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Upper_Bound'], mode='lines', name='Upper Bound', line=dict(dash='dot', color='green')))
    fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Lower_Bound'], mode='lines', name='Lower Bound', line=dict(dash='dot', color='red')))

    fig.update_layout(title="Aluminium Price Forecast", xaxis_title="Date", yaxis_title="Price", template="plotly_white")

    forecast_html = forecast_df.to_html(classes='table table-striped', index=False)
    plot_html = fig.to_html(full_html=False)

    return render_template("data.html", forecast_html=forecast_html, plot_html=plot_html)

if __name__ == "__main__":
    app.run(debug=True, port=5001)