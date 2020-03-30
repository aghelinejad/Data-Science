# Dependencies
from flask import Flask, jsonify, request
from flask_restful import Api, Resource
import joblib

import pandas as pd
import numpy as np

# Your API definition
app = Flask(__name__)
api = Api(app)

# Loading the already persisted model into memory when the application starts
model = joblib.load('RFC_model.pkl')

# Create an API endpoint that takes input variables, transforms them into appropriate format, and returns predictions
class Prediction(Resource):
    @staticmethod
    def post():
        posted_data = request.get_json()
        query = pd.DataFrame([posted_data]).drop('Time', axis=1)
        prediction = list(model.predict(query))
        return jsonify({"prediction": str(prediction)})

api.add_resource(Prediction, '/predict')

if __name__ == '__main__':
    app.run(debug=True)