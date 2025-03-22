# Real-Time-Weather-Forecasting
Built backend ML model to predict "Rain Tomorrow" with weather data (temp, humidity, wind, pressure). Preprocessed using Pandas, applied RandomForestClassifier/Regressor (MSE 0.16). Integrated OpenWeatherMap API for real-time data. Designed for frontend, uses Scikit-Learn/Pandas.

# Weather Prediction Backend Model

A backend machine learning model designed to predict rainfall ("Rain Tomorrow") and forecast temperature/humidity trends using historical and real-time weather data.

# Features
- Predicts rainfall using RandomForestClassifier with a mean squared error of 0.16.
- Forecasts future temperature and humidity using RandomForestRegressor.
- Integrates real-time weather data from the OpenWeatherMap API.
- Processes features like temperature, humidity, wind direction, and pressure.

# Requirements
- Python 3.8+
- Libraries: See `requirements.txt`

# Installation
1. Clone the repository:
 ```bash
git clone https://github.com/yourusername/weather-prediction-backend.git
```
   
2.Install dependencies:

  ```bash
    pip install -r requirements.txt
  ```

3.Obtain an API key from OpenWeatherMap and replace the placeholder in the code.

# Usage
1.Save your historical weather data as weather.csv in the project folder.

2.Run the script:
```bash
    python weather_prediction.py
  ```

3.Enter a city name when prompted to get predictions.


# Project Structure

1.weather_prediction.py: Main backend script with data processing and model logic.

2.requirements.txt: List of required Python libraries.

3.README.md: Project documentation.

# Notes
1.This is a backend-only implementation; frontend integration (e.g., GUI or web app) can be added separately.

2.Replace 'Api key' in the code with your own OpenWeatherMap API key.
