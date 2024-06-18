
import streamlit as st
from datetime import datetime
import joblib

# Load your trained ARIMA model
model = joblib.load(r'D:\2024\Liuis\codes\Pernoctaciones data cleaning\my_arima_model.pkl')

def predict(model, periods):
# Function to make predictions using the model
# 'periods' is the number of future periods to predict
    forecast_results = model.get_forecast(steps=periods)
    forecast = forecast_results.predicted_mean
    conf_int = forecast_results.conf_int()
    return forecast, conf_int


st.title('ARIMA Model Prediction')

periods_input = st.number_input('Enter the number of days for prediction:', min_value=1, value=10)


# Button to make predictions
if st.button('Predict'):
    forecast, conf_int = predict(model, periods_input)
    st.write('Forecasted Values:')
    st.write(forecast)
    st.write('Confidence Intervals:')
    st.write(conf_int)