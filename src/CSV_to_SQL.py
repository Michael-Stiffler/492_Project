
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
import os
import json

# for splitting the data into train and test samples
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report  # for model evaluation metrics
from sklearn.preprocessing import MinMaxScaler  # for feature scaling
# to encode categorical variables
from sklearn.preprocessing import OrdinalEncoder
from sklearn.neighbors import KNeighborsClassifier  # for KNN classification
from sklearn.neighbors import KNeighborsRegressor  # for KNN regression

# import plotly.express as px # for data visualization
import pyodbc
from sqlite3 import Cursor

# database = "Database here"
# server = "Sever name here"

database = "Stiffler_DB"
server = 'DESKTOP-2N4AS7M\SQLEXPRESS'

# database = "WILLS_DB"
# server = 'DESKTOP-TUJPIMN'


def add_marker_to_SQL(formatted_date, lat_in, long_in,
                      city, state, country, zipcode, result):
    connect = pyodbc.connect(
        "DRIVER={SQL Server};SERVER="+server+";Trusted_Connection=yes;")
    connect.autocommit = True
    cursor = connect.cursor()

    # Create database
    createdb = "IF DB_ID('"+database+"') IS NULL CREATE DATABASE "+database+";"
    cursor.execute(createdb)

    # Switches to new database
    switchdb = "USE "+database+";"
    cursor.execute(switchdb)

    cursor.execute("INSERT INTO results (date,lat,lon,city,state,country,zipcode,result) VALUES('{0}','{1}','{2}','{3}','{4}','{5}','{6}','{7}')".format(
        formatted_date, lat_in, long_in, city, state, country, zipcode, result))


def return_results_data():
    data_frame = pd.read_sql('SELECT * FROM results ORDER BY date', connect)
    results_json = json.loads(data_frame.to_json(orient="split"))
    return results_json


def return_model_data():
    return model_data


def return_landslide_json():
    return landslide_json


def connect_sql():
    # Connect to SQL server
    connect = pyodbc.connect(
        "DRIVER={SQL Server};SERVER="+server+";Trusted_Connection=yes;")
    connect.autocommit = True
    cursor = connect.cursor()

    # Create database
    createdb = "IF DB_ID('"+database+"') IS NULL CREATE DATABASE "+database+";"
    cursor.execute(createdb)

    # Switches to new database
    switchdb = "USE "+database+";"
    cursor.execute(switchdb)

    # tableinfo = "CREATE TABLE dbo.landslide(landslide_size VARCHAR(14),fatality_count VARCHAR(10),date VARCHAR(15),country_name VARCHAR(30),cause VARCHAR(30),injuries VARCHAR(30),type VARCHAR(30),longitude float,latitude float,classification int);"
    # cursor.execute(tableinfo)
    return cursor


# connect to SQL server and create database/table
cursor = connect_sql()


# checks if landslide table already exists in the database
# if it does it continues to classification
# if not it creates the table
cursor.execute("""
    SELECT COUNT(*)
    FROM information_schema.tables
    WHERE table_name = '{0}'
    """.format("landslide".replace('\'', '\'\'')))
if cursor.fetchone()[0] == 1:
    print('---------------------------------------------------------')
    print("Using data from SQL")
    print('---------------------------------------------------------')
else:
    print('---------------------------------------------------------')
    print("Importing landslide data from CSV to SQL")
    print('---------------------------------------------------------')
    # prompts user for location of stripped data
    # loc = input("Enter stripped data location:")
    import pathlib
    path_parent = os.path.dirname(pathlib.Path(__file__).parent.resolve())
    os.chdir(path_parent)
    path = os.getcwd()
    path = path + "/src/static/data/Global_Landslide_Catalog_Export_stripped.csv"
    loc = path.replace("/", "\\")
    loc2 = loc.replace(".csv", "2.csv")
    # D:/Downloads/Global_Landslide_Catalog_Export_stripped.csv
    df = pd.read_csv(loc)

    # creates variables
    lat = df["latitude"]
    lon = df["longitude"]
    size = df["landslide_size"]
    fat1 = df["fatality_count"]
    fat = fat1.to_numpy()
    lat = lat.to_numpy()
    lon = lon.to_numpy()
    # makes a coordinate numpy array out of longitude and latitude
    coord = np.column_stack((lon, lat))

    # assigns ints to size of landslides
    landslide1 = df["landslide_size"]
    landslide1 = landslide1.to_numpy()
    size_array = []
    fat_array = []
    for word in landslide1:
        if word == 'small':
            word = 0
            size_array.append(word)
        elif word == 'medium':
            word = 2
            size_array.append(word)
        elif word == 'large':
            word = 4
            size_array.append(word)
        elif word == 'very_large':
            word = 9
            size_array.append(word)
        elif word == 'unknown':
            word = 0
            size_array.append(word)
        else:
            word = 0
            size_array.append(word)

    # assigns int to number of fatalities for classification
    for count in fat:
        print(count)
        num = count
        if num == 0:
            word = 0
            fat_array.append(word)
        elif num > 0 and num < 2:
            word = 2
            fat_array.append(word)
        elif num > 1 and num < 11:
            word = 3
            fat_array.append(word)
        elif num > 11 and num < 31:
            word = 5
            fat_array.append(word)
        elif num > 31:
            word = 7
            fat_array.append(word)
        else:
            word = 0
            fat_array.append(word)

    # combines the two numbers
    i = 0
    class_array = []
    for i in range(len(fat_array)):
        temp = fat_array[i] + size_array[i]
        class_array.append(temp)

    # used the combined two numbers to classify the level of threat
    classification = []
    for i in class_array:
        if i < 2:
            temp = 0
            classification.append(temp)
        elif i > 1 and i < 5:
            temp = 1
            classification.append(temp)
        elif i > 4:
            temp = 2
            classification.append(temp)
        else:
            temp = 0
            classification.append(temp)
        # elif i > 4:
        # temp = "high risk"
        # classification.append(temp)

    from csv import writer
    from csv import reader

    # print(len(classification))
    classification2 = classification
    classification2.insert(0, "classification")

    # Rewrites the data to a new csv with the added classification column
    with open(loc, encoding="utf8") as read_obj, \
            open(loc2, 'w', newline='') as write_obj:
        # Create a csv.reader object from the input file object
        csv_reader = reader(read_obj)
        # Create a csv.writer object from the output file object
        csv_writer = writer(write_obj)
        # Read each row of the input csv file as list
        count = 0
        for row in csv_reader:
            temp = classification[count]
            row.append(temp)
            count = count + 1
            # Add the updated row / list to the output file
            csv_writer.writerow(row)

    # creates new table 'landslide' in WAMP SQL server
    tableinfo = "CREATE TABLE dbo.landslide(landslide_size VARCHAR(14),fatality_count VARCHAR(10),date VARCHAR(60),country_name VARCHAR(60),cause VARCHAR(60),injuries VARCHAR(60),type VARCHAR(60),longitude float,latitude float,classification int);"
    cursor.execute(tableinfo)

    df2 = pd.read_csv(loc2)
    # sends new data csv to WAMP SQL server
    for index, row in df2.iterrows():
        cursor.execute("INSERT INTO landslide (landslide_size,fatality_count,date,country_name,cause,injuries,type,longitude,latitude,classification) VALUES('{0}','{1}','{2}','{3}','{4}','{5}','{6}','{7}','{8}','{9}')".format(
            row.landslide_size, row.fatality_count, row.date, row.country_name, row.landslide_trigger, row.injuries, row.type, row.longitude, row.latitude, row.classification))

    # deletes the second CSV and continues to classification
    import os
    os.remove(loc2)


# Connect to SQL server
connect = pyodbc.connect(
    "DRIVER={SQL Server};SERVER="+server+";Trusted_Connection=yes;")
connect.autocommit = True
cursor = connect.cursor()

# Switches to new database
switchdb = "USE "+database+";"
cursor.execute(switchdb)

# reads in the table from the databse
df4 = pd.read_sql('SELECT * FROM landslide', connect)
landslide_json = json.loads(df4.to_json(orient="split"))

# redoes the coordinates for nearest neighbor
lat = df4["latitude"]
lon = df4["longitude"]
lat = lat.to_numpy()
lon = lon.to_numpy()

coord = np.column_stack((lat, lon))


# Do Min-Max scaling
scaler = MinMaxScaler()

df4['latitude scl'] = scaler.fit_transform(df4[['latitude']])
df4['longitude scl'] = scaler.fit_transform(df4[['longitude']])

# Split into train and test dataframes
df_train, df_test = train_test_split(df4, test_size=0.3)

# independent varibales
X_train2 = df_train[['latitude scl', 'longitude scl']]
X_test2 = df_test[['latitude scl', 'longitude scl']]

# Target for classification model
yC_train = df_train['classification'].ravel()
yC_test = df_test['classification'].ravel()

# Target for regression model
yR_train = df_train['classification'].ravel()
yR_test = df_test['classification'].ravel()

# ---------- Step 3a - Set model parameters - Classification
modelC = KNeighborsClassifier(n_neighbors=5,  # default=5
                              weights='distance',
                              algorithm='ball_tree',
                              metric_params=None,
                              n_jobs=-1
                              )

modelR = KNeighborsRegressor(n_neighbors=10,  # default=5
                             # {‘uniform’, ‘distance’} or callable, default='uniform'
                             weights='uniform',
                             # {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, default=’auto’
                             algorithm='auto',
                             # leaf_size=30, #default=30, Leaf size passed to BallTree or KDTree.
                             # p=2, #default=2, Power parameter for the Minkowski metric.
                             # metric='minkowski', #default=’minkowski’, with p=2 is equivalent to the standard Euclidean metric.
                             # dict, default=None, Additional keyword arguments for the metric function.
                             metric_params=None,
                             n_jobs=-1  # default=None, The number of parallel jobs to run for neighbors search, -1 means using all processors
                             )


# ---------- Step 4 - Fit the two models
clf = modelC.fit(X_train2, yC_train)
reg = modelR.fit(X_train2, yR_train)
reg2 = modelR.fit(X_train2, yR_train)


# ---------- Step 5 - Predict class labels / target values
# Predict on training data
pred_labels_tr = modelC.predict(X_train2)
# pred_values_tr = modelR.predict(X_train)
pred_values_tr2 = modelR.predict(X_train2)

# Predict on a test data
pred_labels_te = modelC.predict(X_test2)
# pred_values_te = modelR.predict(X_test)
pred_values_te2 = modelR.predict(X_test2)


print("Done!")

# # ---------- Step 6 - Print model results
# # Basic info about the model
# print('---------------------------------------------------------')
# print('****************** KNN Classification ******************')
# print('Classes: ', clf.classes_)
# print('Effective Metric: ', clf.effective_metric_)
# print('Effective Metric Params: ', clf.effective_metric_params_)
# print('No. of Samples Fit: ', clf.n_samples_fit_)
# print('Outputs 2D: ', clf.outputs_2d_)
# print('--------------------------------------------------------')
# print("")

print('*************** Evaluation on Test Data ***************')
scoreC_te = modelC.score(X_test2, yC_test)
print('Accuracy Score: ', scoreC_te)
# Look at classification report to evaluate the model
print(classification_report(yC_test, pred_labels_te))

report = classification_report(yC_test, pred_labels_te, output_dict=True)
df = pd.DataFrame(report).transpose()
result = df.to_json(orient="split")
model_data = json.loads(result)
model_data["accuracy"] = scoreC_te
print(model_data)
print('--------------------------------------------------------')
print("")

# # print('*************** Evaluation on Training Data ***************')
# # scoreC_tr = modelC.score(X_train2, yC_train)
# # print('Accuracy Score: ', scoreC_tr)
# # # Look at classification report to evaluate the model
# # print(classification_report(yC_train, pred_labels_tr))
# # print('---------------------------------------------------------')


# # Basic info about the model
# print("")
# print('****************** KNN Regression ******************')
# print('Effective Metric: ', reg2.effective_metric_)
# print('Effective Metric Params: ', reg2.effective_metric_params_)
# print('No. of Samples Fit: ', reg2.n_samples_fit_)
# print("")
# scoreR_te = modelR.score(X_test2, yR_test)
# print('Test Accuracy Score: ', scoreR_te)
# scoreR_tr = modelR.score(X_train2, yR_train)
# print('Training Accuracy Score: ', scoreR_tr)

# print('---------------------------------------------------------')
