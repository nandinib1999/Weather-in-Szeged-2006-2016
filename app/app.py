from flask import Flask, request
import joblib
import lightgbm
import pandas as pd
import numpy as np

scaler = joblib.load('static/scaler.pkl')
vectorizer_precip = joblib.load('static/vectorizer_precip.pkl')
vectorizer_summary = joblib.load('static/vectorizer_summary.pkl')
pca_file = joblib.load('static/pca.pkl')
encoder = joblib.load('static/encoder.pkl')
discretizer = joblib.load('static/discretizer.pkl')
model = lightgbm.Booster(model_file='static/model_lightgbm.pkl')

app = Flask(__name__)

@app.route('/predict-temperature', methods=['POST'])
def predict_temp():
    json_data = request.get_json()
    humidity = 0
    wind_speed = 0
    visibility = 0
    pressure = 0
    month = None
    precip_type = None
    summary = None
    wind_bearing = 0

    if 'humidity' in json_data:
        humidity = json_data['humidity']
    if 'wind_speed' in json_data:
        wind_speed = json_data['wind_speed']
    if 'visibility' in json_data:
        visibility = json_data['visibility']
    if 'pressure' in json_data:
        pressure = json_data['pressure']
    month = json_data['month']
    precip_type = json_data['precip_type']
    summary = json_data['summary']
    if 'wind_bearing' in json_data:
        wind_bearing = json_data['wind_bearing']

    df = pd.DataFrame([[humidity, wind_speed, visibility, pressure, month, precip_type, summary, wind_bearing]], columns=['Humidity', 'Wind Speed (km/h)', 'Visibility (km)', 'Pressure (millibars)', 'Month', 'Precip_type', 'Summary', 'Wind Bearing (degrees)'])
    
    vector_precip = vectorizer_precip.transform(df['Precip_type'])
    vector_precip = vector_precip.toarray()
    for feat_index, feat_name in enumerate(vectorizer_precip.get_feature_names_out()):
        df.loc[0, feat_name] = vector_precip[:, feat_index]

    vector_summary = vectorizer_summary.transform(df['Summary'])
    vector_summary = vector_summary.toarray()
    for feat_index, feat_name in enumerate(vectorizer_summary.get_feature_names_out()):
        df.loc[0, feat_name] = vector_summary[:, feat_index]

    features_to_be_scaled = ['Humidity','Wind Speed (km/h)','Wind Bearing (degrees)','Visibility (km)','Pressure (millibars)']
    scaled_data = scaler.transform(df[features_to_be_scaled])
    df[features_to_be_scaled] = scaled_data
    
    df['Wind Bearing (degrees)'] = discretizer.transform(df[['Wind Bearing (degrees)']])
    df['Wind_Bearing_cat'] = encoder.fit_transform(df[['Wind Bearing (degrees)']])

    df_final = df[['Humidity', 'Wind Speed (km/h)', 'Visibility (km)', 'Pressure (millibars)', 'Month', 'breezy', 'clear', 'cloudy', 'dangerously', 'drizzle', 'dry', 'foggy', 'humid', 'light', 'mostly', 'overcast', 'partly', 'rain', 'windy', 'snow', 'Wind_Bearing_cat']]

    pca_df = pca_file.transform(df_final)

    predicted_temperature = model.predict(pca_df)
    predicted_temperature = predicted_temperature.tolist()

    return {'response':predicted_temperature[0]}

if __name__ == '__main__':
    app.debug = True
    app.run()
