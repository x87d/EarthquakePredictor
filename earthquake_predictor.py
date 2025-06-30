import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import sys

# Load both CSV files and concatenate
files = ['earthquake_data.csv', 'earthquake_1995-2023.csv']
dfs = []
for file in files:
    try:
        df = pd.read_csv(file)
        dfs.append(df)
    except Exception as e:
        print(f"Error loading {file}: {e}")
        sys.exit(1)

data = pd.concat(dfs, ignore_index=True)

# Clean and preprocess data
def preprocess(df):
    # Standardize date_time
    df = df.copy()
    df['date_time'] = pd.to_datetime(df['date_time'], errors='coerce', dayfirst=True)
    df = df.dropna(subset=['date_time', 'magnitude'])
    df['magnitude'] = pd.to_numeric(df['magnitude'], errors='coerce')
    df = df.dropna(subset=['magnitude'])
    # Fill missing country/region/continent with 'Unknown'
    for col in ['country', 'continent']:
        if col not in df.columns:
            df[col] = 'Unknown'
        else:
            df[col] = df[col].fillna('Unknown')
    df = df.sort_values('date_time')
    return df

data = preprocess(data)

# Count records per country, continent, and globally
country_counts = data['country'].value_counts()
continent_counts = data['continent'].value_counts()
global_count = len(data)

MIN_RECORDS = 5

# Show available options
print("Countries with sufficient data:")
for c, n in country_counts.items():
    if n >= MIN_RECORDS:
        print(f"  {c} ({n})")
print("\nContinents with sufficient data:")
for c, n in continent_counts.items():
    if n >= MIN_RECORDS:
        print(f"  {c} ({n})")
print(f"\nOr type 'global' for all data ({global_count} records)")

# User input
user_input = input("Enter a country, continent, or 'global': ").strip()

# Data selection logic
if user_input.lower() == 'global':
    selected_data = data.copy()
    label = 'Global'
    count = global_count
elif user_input in country_counts and country_counts[user_input] >= MIN_RECORDS:
    selected_data = data[data['country'] == user_input]
    label = user_input
    count = country_counts[user_input]
elif user_input in continent_counts and continent_counts[user_input] >= MIN_RECORDS:
    selected_data = data[data['continent'] == user_input]
    label = user_input
    count = continent_counts[user_input]
else:
    # Fallback: try region/continent/global
    fallback = None
    if user_input in country_counts:
        country = user_input
        continent = data[data['country'] == country]['continent'].mode().values[0]
        if continent in continent_counts and continent_counts[continent] >= MIN_RECORDS:
            fallback = ('continent', continent, continent_counts[continent])
    if not fallback and global_count >= MIN_RECORDS:
        fallback = ('global', 'Global', global_count)
    if fallback:
        print(f"\nNot enough data for '{user_input}'. Using {fallback[0]}: {fallback[1]} ({fallback[2]} records)")
        if fallback[0] == 'continent':
            selected_data = data[data['continent'] == fallback[1]]
            label = fallback[1]
            count = fallback[2]
        else:
            selected_data = data.copy()
            label = 'Global'
            count = global_count
    else:
        print(f"Not enough data for '{user_input}' and no suitable fallback found.")
        sys.exit(1)

selected_data = selected_data.sort_values('date_time').reset_index(drop=True)

if len(selected_data) < MIN_RECORDS:
    print(f"Not enough data for prediction after filtering. Exiting.")
    sys.exit(1)

# Feature engineering: intervals and previous magnitudes
selected_data['interval_years'] = selected_data['date_time'].diff().dt.total_seconds() / (365.25*24*3600)
selected_data['prev_magnitude'] = selected_data['magnitude'].shift(1)
selected_data = selected_data.dropna(subset=['interval_years', 'prev_magnitude'])

# Prepare features and targets
X = selected_data[['interval_years', 'prev_magnitude']].shift(1).dropna()
y_interval = selected_data['interval_years'][1:]
y_magnitude = selected_data['magnitude'][1:]

X = X.iloc[:-1]
y_interval = y_interval.iloc[1:]
y_magnitude = y_magnitude.iloc[1:]

if len(X) < MIN_RECORDS:
    print(f"Not enough feature data for prediction. Exiting.")
    sys.exit(1)

# Train/test split
X_train, X_test, y_interval_train, y_interval_test = train_test_split(X, y_interval, test_size=0.2, random_state=42)
_, _, y_magnitude_train, y_magnitude_test = train_test_split(X, y_magnitude, test_size=0.2, random_state=42)

# Regression models
interval_model = RandomForestRegressor(n_estimators=100, random_state=42)
magnitude_model = RandomForestRegressor(n_estimators=100, random_state=42)
interval_model.fit(X_train, y_interval_train)
magnitude_model.fit(X_train, y_magnitude_train)

# Predict next interval and magnitude
last_row = selected_data.iloc[-1]
last_features = np.array([[last_row['interval_years'], last_row['prev_magnitude']]])
predicted_interval = interval_model.predict(last_features)[0]
predicted_magnitude = magnitude_model.predict(last_features)[0]

print(f"\nPredictions for {label} ({count} records used):")
if count < 15:
    print("(Warning: Prediction may be unreliable due to limited data.)")
print(f"Next earthquake expected in {predicted_interval:.2f} years from now")
print(f"Predicted magnitude: {predicted_magnitude:.2f}") 