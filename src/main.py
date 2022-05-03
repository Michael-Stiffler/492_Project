from flask import Flask, render_template, redirect, url_for, request, make_response, jsonify
from flask import make_response
from waitress import serve

from math import sqrt
from re import search
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn import neighbors, datasets

from sklearn import datasets
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

import pandas as pd  # for data manipulation
import numpy as np  # for data manipulation

# for splitting the data into train and test samples
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report  # for model evaluation metrics
from sklearn.preprocessing import MinMaxScaler  # for feature scaling
# to encode categorical variables
from sklearn.preprocessing import OrdinalEncoder
from sklearn.neighbors import KNeighborsClassifier  # for KNN classification
from sklearn.neighbors import KNeighborsRegressor  # for KNN regression

import pyodbc
from sqlite3 import Cursor
from sqlite3 import connect
from CSV_to_SQL import coord, modelR, modelC


# assesses the final number and assigns a risk level
def assess(result, var):
    result[0] = result[0] - var
    print(result[0])
    if result[0] <= 0.4999999:
        return "low risk"
    elif result[0] >= 0.5 and result[0] <= 1.09999999:
        return "medium risk"
    elif result[0] >= 1.1:
        return "high risk"


# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
    distance = 0.0
    x = np.array2string(row2)
    # print(x)
    x = x[1:-1]
    temp = []
    temp = x.split()
    # print(temp)
    for i in range(len(row1)-1):
        temp2 = float(temp[i])
        distance += (row1[i] - temp2)**2
    return sqrt(distance)


# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors


app = Flask(__name__)


@app.route("/")
def index():
    return render_template('/index.html')


@app.route("/predict")
def predict():
    return render_template('/predict.html')


@app.route("/sandbox")
def sandbox():
    return render_template('/sandbox.html')


@app.route('/datapull', methods=['GET', 'POST'])
def datapull():
    if request.method == 'POST':
        datafromjs = request.form['data']

        x = datafromjs.split("#", 1)

        lat_in = float(x[0])
        long_in = float(x[1])

        # for row in coord:
        #     distance = euclidean_distance([lat_in, long_in], row)
        #     # print(row)
        #     # print(distance)

        neighbors = get_neighbors(coord, [lat_in, long_in], 3)
        var = 0
        for neighbor in neighbors:
            # print(neighbor)
            near = neighbor
            # print(abs(long_in - neighbor[1]))
            if (abs(lat_in - neighbor[0]) > 20) or (abs(long_in - neighbor[1]) > 20):
                var = var + .15
            elif (abs(lat_in - neighbor[0]) > 40) or (abs(long_in - neighbor[1]) > 40):
                var = var + .15
            elif abs(lat_in - neighbor[0]) < 15 and abs(long_in - neighbor[1]) < 15:
                var = var - .2
            else:
                var = var
            # print(var)

        result = modelC.predict([[lat_in, long_in]])
        fin = assess(result, var)

        resp = jsonify(fin)
        return resp


if __name__ == "__main__":
    serve(app, host='0.0.0.0', port=8000)
