"""

    Simple Flask-based API for Serving an sklearn Model.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file instantiates a Flask webserver
    as a means to create a simple API used to deploy models trained within
    the sklearn framework.

"""

# API Dependencies
import pickle
import json
import numpy as np
from model import load_model, make_prediction
from flask import Flask, request, jsonify

# Application definition
app = Flask(__name__)

# Load our model into memory.
# Please update this path to reflect your own trained model.
static_model = load_model(
    path_to_model='assets/trained-models/rdf_model_new.pkl')

print ('-'*40)
print ('Model successfully loaded')
print ('-'*40)

""" You may use this section (above the app routing function) of the python script to implement 
    any auxiliary functions required to process your model's artifacts.
"""


# Define the API's interface.
# Here the 'model_prediction()' function will be called when a POST request
# is sent to our interface located at:
# http:{Host-machine-ip-address}:5000/api_v0.1
@app.route('/api_v0.1', methods=['POST'])
def model_prediction():
    # We retrieve the data payload of the POST request
    data = request.get_json(force=False)
    print(data)
    # We then preprocess our data, and use our pretrained model to make a
    # prediction.


    output = make_prediction(data, static_model)
    # We finally package this prediction as a JSON object to deliver a valid
    # response with our API.
    return jsonify(output)

# Configure Server Startup properties.
# Note:
# When developing your API, set `debug=True`
# This will allow Flask to automatically restart itself everytime you
# update your API code.
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
    # input = {"Unnamed: 0":8764,"time":"2018-01-01 03:00:00","Madrid_wind_speed":4.6666666667,"Valencia_wind_deg":"level_8","Bilbao_rain_1h":0.0,"Valencia_wind_speed":5.3333333333,"Seville_humidity":89.0,"Madrid_humidity":78.0,"Bilbao_clouds_all":0.0,"Bilbao_wind_speed":3.6666666667,"Seville_clouds_all":0.0,"Bilbao_wind_deg":143.3333333333,"Barcelona_wind_speed":4.6666666667,"Barcelona_wind_deg":266.6666666667,"Madrid_clouds_all":0.0,"Seville_wind_speed":0.6666666667,"Barcelona_rain_1h":0.0,"Seville_pressure":"sp25","Seville_rain_1h":0.0,"Bilbao_snow_3h":0,"Barcelona_pressure":1020.3333333333,"Seville_rain_3h":0.0,"Madrid_rain_1h":0.0,"Barcelona_rain_3h":0.0,"Valencia_snow_3h":0,"Madrid_weather_id":800.0,"Barcelona_weather_id":800.3333333333,"Bilbao_pressure":1026.6666666667,"Seville_weather_id":800.0,"Valencia_pressure":null,"Seville_temp_max":282.4833333333,"Madrid_pressure":1030.3333333333,"Valencia_temp_max":284.15,"Valencia_temp":284.15,"Bilbao_weather_id":721.0,"Seville_temp":281.6733333333,"Valencia_humidity":53.6666666667,"Valencia_temp_min":284.15,"Barcelona_temp_max":284.8166666667,"Madrid_temp_max":280.4833333333,"Barcelona_temp":284.19,"Bilbao_temp_min":277.8166666667,"Bilbao_temp":281.01,"Barcelona_temp_min":283.4833333333,"Bilbao_temp_max":284.15,"Seville_temp_min":281.15,"Madrid_temp":279.1933333333,"Madrid_temp_min":278.15}
    # result =  make_prediction(input, static_model)
