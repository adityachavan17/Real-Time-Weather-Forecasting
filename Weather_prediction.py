


import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import pytz
import pymysql
from dotenv import load_dotenv
import os



load_dotenv()

API_KEY = os.getenv("API_KEY")
DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME") 
BASE_URL = 'https://api.openweathermap.org/data/2.5/'


def get_current_weather(city):
    url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()

    if 'name' not in data:
        print(f"Error fetching weather data: {data.get('message', 'Unknown Error')}")
        return None

    return {
        'city': data['name'],
        'current_temp': round(data['main']['temp']),
        'feels_like': round(data['main']['feels_like']),
        'temp_min': round(data['main']['temp_min']),
        'temp_max': round(data['main']['temp_max']),
        'humidity': round(data['main']['humidity']),
        'description': data['weather'][0]['description'],
        'country': data['sys']['country'],
        'wind_gust_dir': data['wind'].get('deg', 0),   
        'pressure': data['main']['pressure'],
        'wind_gust_speed': data['wind'].get('speed', 0),
        'record_time': datetime.now()  
    }


def prepare_data(data):
    le_wind = LabelEncoder()
    le_rain = LabelEncoder()

    data_clean = data.dropna(subset=['WindGustDir', 'RainTomorrow']).copy()
    data_clean['WindGustDir'] = le_wind.fit_transform(data_clean['WindGustDir'])
    data_clean['RainTomorrow'] = le_rain.fit_transform(data_clean['RainTomorrow'])

    X = data_clean[['MinTemp', 'MaxTemp', 'WindGustDir',
                    'WindGustSpeed', 'Humidity', 'Pressure', 'Temp']]
    y = data_clean['RainTomorrow']
    return X, y, le_wind


def train_rain_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Mean squared error for rain model:", mean_squared_error(y_test, y_pred))
    return model


def prepare_regression_data(data, feature):
    X, y = [], []
    for i in range(len(data) - 1):
        X.append(data[feature].iloc[i])
        y.append(data[feature].iloc[i + 1])
    return np.array(X).reshape(-1, 1), np.array(y)


def train_regression_model(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model


def predict_future(model, current_value):
    predictions = [current_value]
    for _ in range(5):
        next_value = model.predict(np.array([[predictions[-1]]]))
        predictions.append(next_value[0])
    return predictions[1:]


def create_mysql_connection():
    try:
        conn = pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
        return conn
    except Exception as e:
        print(f"Failed to connect to MySQL: {e}")
        return None


def create_tables(conn):
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS weather_records (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    city VARCHAR(50),
                    country VARCHAR(10),
                    record_time DATETIME,
                    current_temp FLOAT,
                    feels_like FLOAT,
                    temp_min FLOAT,
                    temp_max FLOAT,
                    humidity INT,
                    description VARCHAR(100),
                    wind_gust_dir FLOAT,
                    pressure INT,
                    wind_gust_speed FLOAT,
                    rain_prediction VARCHAR(10),
                    actual_rain VARCHAR(10) DEFAULT NULL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS weather_forecast (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    record_id INT,
                    forecast_time DATETIME,
                    predicted_temp FLOAT,
                    predicted_humidity FLOAT,
                    FOREIGN KEY (record_id) REFERENCES weather_records(id)
                )
            """)
            conn.commit()
    except Exception as e:
        print(f"Error creating tables: {e}")


def insert_weather_record(conn, record):
    with conn.cursor() as cursor:
        sql = """
            INSERT INTO weather_records 
            (city, country, record_time, current_temp, feels_like, temp_min, temp_max,
             humidity, description, wind_gust_dir, pressure, wind_gust_speed, rain_prediction)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(sql, (
            record['city'], record['country'], record['record_time'],
            record['current_temp'], record['feels_like'], record['temp_min'],
            record['temp_max'], record['humidity'], record['description'],
            record['wind_gust_dir'], record['pressure'], record['wind_gust_speed'],
            record['rain_prediction']
        ))
        conn.commit()
        return cursor.lastrowid


def insert_forecasts(conn, record_id, forecasts):
    with conn.cursor() as cursor:
        sql = """
            INSERT INTO weather_forecast 
            (record_id, forecast_time, predicted_temp, predicted_humidity)
            VALUES (%s, %s, %s, %s)
        """
        cursor.executemany(sql, [
            (record_id, ft, t, h) for ft, t, h in forecasts
        ])
        conn.commit()


def calculate_rain_accuracy(conn):
    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT rain_prediction, actual_rain 
            FROM weather_records
            WHERE record_time >= NOW() - INTERVAL 7 DAY AND actual_rain IS NOT NULL
        """)
        rows = cursor.fetchall()

    if not rows:
        return 0.0, 0

    correct = sum(1 for pred, actual in rows if pred == actual)
    accuracy = (correct / len(rows)) * 100
    return accuracy, len(rows)


def weather_view():
    city = input("Enter city Name: ").strip()
    current_weather = get_current_weather(city)
    if current_weather is None:
        return

    historical_data = pd.read_csv('weather.csv')
    X, y, le_wind = prepare_data(historical_data)
    rain_model = train_rain_model(X, y)

    wind_deg = current_weather['wind_gust_dir'] % 360
    compass_points = [
        ("N", 0, 11.25), ("NNE", 11.25, 33.75), ("NE", 33.75, 56.25),
        ("ENE", 56.25, 78.75), ("E", 78.75, 101.25), ("ESE", 101.25, 123.75),
        ("SE", 123.75, 146.25), ("SSE", 146.25, 168.75), ("S", 168.75, 191.25),
        ("SSW", 191.25, 213.75), ("SW", 213.75, 236.25), ("WSW", 236.25, 258.75),
        ("W", 258.75, 281.25), ("WNW", 281.25, 303.75), ("NW", 303.75, 326.25),
        ("NNW", 326.25, 348.75)
    ]
    compass_direction = next((point for point, start, end in compass_points if start <= wind_deg < end), "N")
    compass_direction_encoded = le_wind.transform([compass_direction])[0] if compass_direction in le_wind.classes_ else 0

    current_df = pd.DataFrame([{
        'MinTemp': current_weather['temp_min'],
        'MaxTemp': current_weather['temp_max'],
        'WindGustDir': compass_direction_encoded,
        'WindGustSpeed': current_weather['wind_gust_speed'],
        'Humidity': current_weather['humidity'],
        'Pressure': current_weather['pressure'],
        'Temp': current_weather['current_temp'],
    }])

    rain_prediction = "Yes" if rain_model.predict(current_df)[0] else "No"

    X_temp, y_temp = prepare_regression_data(historical_data, 'Temp')
    X_hum, y_hum = prepare_regression_data(historical_data, 'Humidity')

    temp_model = train_regression_model(X_temp, y_temp)
    hum_model = train_regression_model(X_hum, y_hum)

    future_temp = predict_future(temp_model, current_weather['temp_min'])
    future_humidity = predict_future(hum_model, current_weather['humidity'])

    timezone = pytz.timezone('Asia/Kolkata')
    now = datetime.now(timezone)
    next_hour = now + timedelta(hours=1)
    next_hour = next_hour.replace(minute=0, second=0, microsecond=0)
    future_times = [(next_hour + timedelta(hours=i)) for i in range(5)]

    print(f'\nCity: {city}, {current_weather["country"]}')
    print(f'Current Temperature: {current_weather["current_temp"]}°C')
    print(f'Feels like: {current_weather["feels_like"]}°C')
    print(f'Minimum Temperature: {current_weather["temp_min"]}°C')
    print(f'Maximum Temperature: {current_weather["temp_max"]}°C')
    print(f'Humidity: {current_weather["humidity"]}%')
    print(f'Weather Description: {current_weather["description"]}')
    print(f'Rain Prediction: {rain_prediction}')

    print("\nFuture Temperature Predictions:")
    for time, temp in zip(future_times, future_temp):
        print(f"{time.strftime('%H:%M')}: {round(temp, 1)}°C")

    print("\nFuture Humidity Predictions:")
    for time, humidity in zip(future_times, future_humidity):
        print(f"{time.strftime('%H:%M')}: {round(humidity, 1)}%")

    conn = create_mysql_connection()
    if conn:
        create_tables(conn)

        record = {
            'city': city,
            'country': current_weather['country'],
            'record_time': current_weather['record_time'],
            'current_temp': current_weather['current_temp'],
            'feels_like': current_weather['feels_like'],
            'temp_min': current_weather['temp_min'],
            'temp_max': current_weather['temp_max'],
            'humidity': current_weather['humidity'],
            'description': current_weather['description'],
            'wind_gust_dir': current_weather['wind_gust_dir'],
            'pressure': current_weather['pressure'],
            'wind_gust_speed': current_weather['wind_gust_speed'],
            'rain_prediction': rain_prediction
        }

        record_id = insert_weather_record(conn, record)

        forecasts = [(ft, round(t, 1), round(h, 1)) for ft, t, h in zip(future_times, future_temp, future_humidity)]
        insert_forecasts(conn, record_id, forecasts)

        accuracy, total = calculate_rain_accuracy(conn)
        print(f"\nRain Prediction Accuracy (last 7 days): {accuracy:.2f}% over {total} records.")

        conn.close()


if __name__ == "__main__":
    weather_view()
