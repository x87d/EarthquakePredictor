# Earthquake Predictor

This project predicts the next earthquake interval and magnitude for a given country, continent, or globally using historical earthquake data and regression models.

## Features
- Loads and merges earthquake data from two CSV files
- Cleans and preprocesses the data
- Trains regression models to predict the next earthquake's interval (years) and magnitude
- **User can select a country, continent, or 'global'** for prediction
- **Script displays available countries and continents with sufficient data**
- **Automatic fallback to continent or global data** if the selected country has too few records
- **Displays the number of records used and warnings** if prediction is based on limited data
- Simple command-line interface (CLI)

## Data Source
This project uses data from the Kaggle dataset:
[Earthquake Dataset by warcoder](https://www.kaggle.com/datasets/warcoder/earthquake-dataset)

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Ensure the following files are in the project directory:
   - `earthquake_data.csv`
   - `earthquake_1995-2023.csv`

## Usage
Run the predictor script:
```bash
python earthquake_predictor.py
```
- The script will display a list of countries and continents with enough data.
- Enter a country, continent, or 'global' when prompted.
- If your selection has too little data, the script will automatically use a broader region (continent or global) and notify you.
- The script will output the predicted years until the next earthquake and its expected magnitude, along with the number of records used and a warning if the prediction is less reliable.

## Notes
- The model requires at least 5 historical earthquakes for a region to make predictions.
- Predictions are based on patterns in the historical data and are for demonstration purposes only. 
