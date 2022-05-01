from math import sqrt
from re import search
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
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

import matplotlib.pyplot as plt  # for data visualization
# import plotly.express as px # for data visualization
# import plotly.express as px # for data visualization
import pyodbc
from sqlite3 import Cursor
from sqlite3 import connect

# Connect to SQL server
database = "WILLS_DB"
server = 'DESKTOP-TUJPIMN'
connect = pyodbc.connect(
    "DRIVER={SQL Server};SERVER="+server+";Trusted_Connection=yes;")
connect.autocommit = True
cursor = connect.cursor()

# Switches to new database
switchdb = "USE "+database+";"
cursor.execute(switchdb)

# reads in the table from the databse
df4 = pd.read_sql('SELECT * FROM landslide', connect)

# redoes the coordinates for nearest neighbor
lat = df4["latitude"]
lon = df4["longitude"]
lat = lat.to_numpy()
lon = lon.to_numpy()

coord = np.column_stack((lon, lat))


# Do Min-Max scaling
scaler = MinMaxScaler()

df4['latitude scl'] = scaler.fit_transform(df4[['latitude']])
df4['longitude scl'] = scaler.fit_transform(df4[['longitude']])

# Split into train and test dataframes
df_train, df_test = train_test_split(df4, test_size=0.2, random_state=42)

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
#pred_values_tr = modelR.predict(X_train)
pred_values_tr2 = modelR.predict(X_train2)

# Predict on a test data
pred_labels_te = modelC.predict(X_test2)
#pred_values_te = modelR.predict(X_test)
pred_values_te2 = modelR.predict(X_test2)


# ---------- Step 6 - Print model results
# Basic info about the model
# print('---------------------------------------------------------')
#print('****************** KNN Classification ******************')
#print('Classes: ', clf.classes_)
#print('Effective Metric: ', clf.effective_metric_)
#print('Effective Metric Params: ', clf.effective_metric_params_)
#print('No. of Samples Fit: ', clf.n_samples_fit_)
#print('Outputs 2D: ', clf.outputs_2d_)
# print('--------------------------------------------------------')
# print("")

#print('*************** Evaluation on Test Data ***************')
#scoreC_te = modelC.score(X_test2, yC_test)
#print('Accuracy Score: ', scoreC_te)
# Look at classification report to evaluate the model
#print(classification_report(yC_test, pred_labels_te))
# print('--------------------------------------------------------')
# print("")

#print('*************** Evaluation on Training Data ***************')
#scoreC_tr = modelC.score(X_train2, yC_train)
#print('Accuracy Score: ', scoreC_tr)
# Look at classification report to evaluate the model
#print(classification_report(yC_train, pred_labels_tr))
# print('---------------------------------------------------------')


# Basic info about the model
# print("")
#print('****************** KNN Regression ******************')
#print('Effective Metric: ', reg2.effective_metric_)
#print('Effective Metric Params: ', reg2.effective_metric_params_)
#print('No. of Samples Fit: ', reg2.n_samples_fit_)
# print("")
#scoreR_te = modelR.score(X_test2, yR_test)
#print('Test Accuracy Score: ', scoreR_te)
#scoreR_tr = modelR.score(X_train2, yR_train)
#print('Training Accuracy Score: ', scoreR_tr)

# print('---------------------------------------------------------')

# assesses the final number and assigns a risk level
def assess(result, var):
    result[0] = result[0] - var
    print(result[0])
    if result[0] <= 0.4999999:
        print("low risk")
    elif result[0] >= 0.5 and result[0] <= 1.19999999:
        print("medium risk")
    elif result[0] >= 1.2:
        print("high risk")


# prompts user for input for latitude and longitude
lat_in = input("Enter latitude : ")
long_in = input("Enter longitude :")


lat_in = float(lat_in)
long_in = float(long_in)


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


for row in coord:
    distance = euclidean_distance([lat_in, long_in], row)
    # print(row)
    # print(distance)

# sees if nearest neighbor is more than 20 degrees away than user inputted location
neighbors = get_neighbors(coord, [lat_in, long_in], 1)
for neighbor in neighbors:
    print(neighbor)
    if abs(lat_in - neighbor[0]) > 20 or abs(long_in - neighbor[1]) > 20:
        var = 0.5
    else:
        var = 0.0


#print( modelC.predict([[25.6,96]]) )
result = modelR.predict([[lat_in, long_in]])
assess(result, var)
