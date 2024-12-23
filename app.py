from flask import Flask, render_template, request
import requests
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Define functions for fetching data and making predictions
def get_total_races(season):
    url = f'http://ergast.com/api/f1/{season}.json'
    response = requests.get(url)
    season_data = response.json()
    if not season_data['MRData']['RaceTable']['Races']:
        return 0
    total_races = len(season_data['MRData']['RaceTable']['Races'])
    return total_races

def fetch_race_data(season, race_number):
    url = f'http://ergast.com/api/f1/{season}/{race_number}/results.json'
    response = requests.get(url)
    race_data = response.json()
    if not race_data['MRData']['RaceTable']['Races']:
        return pd.DataFrame()
    race_results = race_data['MRData']['RaceTable']['Races'][0]['Results']
    race_df = pd.DataFrame([{
        'driver_id': result['Driver']['driverId'],
        'driver_name': f"{result['Driver']['givenName']} {result['Driver']['familyName']}",
        'constructor_id': result['Constructor']['constructorId'],
        'constructor_name': result['Constructor']['name'],
        'grid_position': result['grid'],
        'laps': result['laps'],
        'status': result['status'],
        'position': result['position'],
        'points': result['points'],
        'fastest_lap_time': result.get('FastestLap', {}).get('Time', {}).get('time', None)
    } for result in race_results])
    race_df['race_number'] = race_number
    return race_df

def fetch_previous_races_data(season, race_number):
    all_race_data = pd.DataFrame()
    for race in range(1, race_number):
        race_data = fetch_race_data(season, race)
        if not race_data.empty:
            all_race_data = pd.concat([all_race_data, race_data], ignore_index=True)
    return all_race_data

def prepare_features(data):
    data['driver_id'] = pd.factorize(data['driver_id'])[0]
    data['constructor_id'] = pd.factorize(data['constructor_id'])[0]
    data['grid_position'] = data['grid_position'].astype(int)
    data['points'] = data['points'].astype(float)
    data['fastest_lap_time'] = data['fastest_lap_time'].fillna('0:00:00')
    data['winner'] = (data['position'] == '1').astype(int)
    X = data[['driver_id', 'constructor_id', 'grid_position', 'points', 'race_number']]
    y = data['winner']
    return X, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def predict_winner(model, season, race_number):
    previous_race_data = fetch_previous_races_data(season, race_number)
    if previous_race_data.empty:
        return None, None
    X, _ = prepare_features(previous_race_data)
    last_race_features = X.mean().values.reshape(1, -1)
    X_pred = pd.DataFrame(last_race_features, columns=X.columns)
    prediction = model.predict(X_pred)
    winner_index = prediction.argmax()
    winner = previous_race_data.iloc[winner_index]['driver_name']
    team = previous_race_data.iloc[winner_index]['constructor_name']
    return winner, team

# Flask Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    season = int(request.form['season'])
    race_number = int(request.form['race_number'])
    total_races = get_total_races(season)
    
    if total_races == 0 or race_number < 1 or race_number > total_races:
        return render_template('result.html', error="Invalid season or race number.")
    
    previous_races_data = fetch_previous_races_data(season, race_number)

    if previous_races_data.empty:
        return render_template('result.html', error="Not enough data to make a prediction.")
    
    X, y = prepare_features(previous_races_data)
    model = train_model(X, y)
    winner, team = predict_winner(model, season, race_number)

    if winner is None or team is None:
        return render_template('result.html', error="Not enough data to make a prediction.")
    
    return render_template('result.html', winner=winner, team=team)

if __name__ == '__main__':
    app.run(debug=True)








