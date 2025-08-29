# 🌦️ Real-Time Weather Forecasting with Rain Prediction

This project is a **Real-Time Weather Forecasting System** that fetches current weather data using the OpenWeather API, predicts rain using a machine learning model (Random Forest), and stores results in a **MySQL database**. It also provides future temperature and humidity predictions, along with rain prediction accuracy over the last 7 days.

---

## 🚀 Features
- Fetches **real-time weather** from OpenWeather API.
- Predicts **rain chances (Yes/No)** using Random Forest.
- Provides **future temperature & humidity forecasts** using regression.
- Stores:
  - Current weather → `weather_records` table
  - Forecasts → `weather_forecast` table
- Tracks **rain prediction accuracy** over the past 7 days.

---

## 📂 Project Structure
```
├── Weather_prediction.py   # Main Python script
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── .env                    # API keys & DB credentials (not shared on GitHub)
├── .gitignore              # Ignore sensitive/unnecessary files
└── weather.csv             # Historical dataset for training (optional)
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the repository
```bash
git clone https://github.com/yourusername/weather-forecasting.git
cd weather-forecasting
```

### 2️⃣ Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate    # On Windows
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Configure environment variables
Create a **`.env`** file in the project root:

```
API_KEY=your_openweather_api_key
DB_HOST=localhost
DB_USER=your_mysql_user
DB_PASSWORD=your_mysql_password
DB_NAME=weather_db
```

⚠️ **Never share your `.env` file on GitHub.**

### 5️⃣ Setup MySQL database
```sql
CREATE DATABASE weather_db;
```

The script will automatically create the necessary tables.

### 6️⃣ Run the program
```bash
python Weather_prediction.py
```

Enter a city name, and you’ll see current weather, predictions, and forecast accuracy.

---

## 🛠️ Requirements
- Python 3.11 / 3.12 / 3.13
- MySQL running locally or remotely
- Python libraries (see `requirements.txt`)

---

## 📊 Example Output
```
Enter city Name: Pune

City: Pune, IN
Current Temperature: 22°C
Feels like: 23°C
Minimum Temperature: 22°C
Maximum Temperature: 22°C
Humidity: 89%
Weather Description: overcast clouds
Rain Prediction: Yes

Future Temperature Predictions:
01:00: 23.3°C
02:00: 18.8°C
03:00: 20.9°C
04:00: 18.2°C
05:00: 16.9°C

Future Humidity Predictions:
01:00: 62.6%
02:00: 59.1%
03:00: 66.9%
04:00: 73.1%
05:00: 39.8%

Rain Prediction Accuracy (last 7 days): 85.71% over 14 records.
```

---

## 📌 Notes
- Ensure you have a valid **OpenWeather API key** from [https://openweathermap.org/api](https://openweathermap.org/api).
- Historical dataset (`weather.csv`) is required for training ML models.
- Tested on **Python 3.13** with:
  - pandas >= 2.2.3
  - numpy >= 2.1.0
  - scikit-learn >= 1.5.0

---

## 🤝 Contribution
Feel free to fork the repo, open issues, and submit pull requests.

