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

import pandas as pd # for data manipulation
import numpy as np # for data manipulation

from sklearn.model_selection import train_test_split # for splitting the data into train and test samples
from sklearn.metrics import classification_report # for model evaluation metrics
from sklearn.preprocessing import MinMaxScaler # for feature scaling
from sklearn.preprocessing import OrdinalEncoder # to encode categorical variables
from sklearn.neighbors import KNeighborsClassifier # for KNN classification
from sklearn.neighbors import KNeighborsRegressor # for KNN regression

import matplotlib.pyplot as plt # for data visualization
#import plotly.express as px # for data visualization

import pymysql
conn = pymysql.connect(host="127.0.0.1", user="root", password='',database="mysql")
cursor = conn.cursor()


cursor = conn.cursor()
#checks if landslide table already exists in the database
#if it does it continues to classification
#if not it creates the table
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
    print("Import landslide data from CSV to SQL")
    print('---------------------------------------------------------')
    #prompts user for location of stripped data
    loc = input ("Enter stripped data location:")
    #D:/Downloads/Global_Landslide_Catalog_Export_stripped.csv
    df = pd.read_csv (loc)

    #creates variables 
    lat = df["latitude"]
    lon = df["longitude"]
    size = df["landslide_size"]
    fat = df["fatality_count"]
    lat = lat.to_numpy()
    lon = lon.to_numpy()
    #makes a coordinate numpy array out of longitude and latitude
    coord = np.column_stack((lon, lat))

    #assigns ints to size of landslides
    landslide1 = df["landslide_size"]
    landslide1 = landslide1.to_numpy()
    size_array = []
    fat_array = []
    for word in landslide1:
    # print(word)
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



    #assigns int to number of fatalities for classification
    for count in fat:
    # print(word)
        if count == 0:
            word = 0
            fat_array.append(word)
        elif count > 0 and count < 2:
            word = 2
            fat_array.append(word)
        elif count > 1 and count < 11:
            word = 3
            fat_array.append(word)
        elif count > 11 and count < 31:
            word = 5
            fat_array.append(word)
        elif count > 31:
            word = 7
            fat_array.append(word)
        else:
            word = 0
            fat_array.append(word)

    #combines the two numbers
    i = 0
    class_array =[]
    for i in range(len(fat_array)):
        temp = fat_array[i] + size_array[i]
        class_array.append(temp)


    #used the combined two numbers to classify the level of threat
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
        #elif i > 4:
        # temp = "high risk"
        # classification.append(temp)


    from csv import writer
    from csv import reader

    #print(len(classification))
    classification2 = classification
    classification2.insert(0, "classification")


    #Rewrites the data to a new csv with the added classification column
    with open('D:/Downloads/Global_Landslide_Catalog_Export_stripped.csv', encoding="utf8") as read_obj, \
            open('D:/Downloads/Global_Landslide_Catalog_Export2.csv', 'w', newline='') as write_obj:
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

    #creates new table 'landslide' in WAMP SQL server
    cursor.execute('''
            CREATE TABLE landslide (
                landslide_size varchar(14),
                fatality_count varchar(10),
                longitude float,
                latitude float,
                classification int
                )
                ''')

    df2 = pd.read_csv (r'D:/Downloads/Global_Landslide_Catalog_Export2.csv')
    #sends new data csv to WAMP SQL server
    for index,row in df2.iterrows():
        cursor.execute("INSERT INTO landslide (landslide_size,fatality_count,longitude,latitude,classification) VALUES('{0}','{1}','{2}','{3}','{4}')".format(row.landslide_size,row.fatality_count,row.longitude,row.latitude,row.classification))
                    
    conn.commit()
    #deletes the second CSV and continues to classification
    import os
    os.remove('D:/Downloads/Global_Landslide_Catalog_Export2.csv')



#reads in the table from the databse
df4 = pd.read_sql_query('SELECT * FROM landslide', conn)

#print(df4)
#redoes the coordinates for nearest neighbor
lat = df4["latitude"]
lon = df4["longitude"]
lat = lat.to_numpy()
lon = lon.to_numpy()

coord = np.column_stack((lon, lat))


# Do Min-Max scaling
scaler = MinMaxScaler()

df4['latitude scl']=scaler.fit_transform(df4[['latitude']])
df4['longitude scl']=scaler.fit_transform(df4[['longitude']])

# Split into train and test dataframes
df_train, df_test = train_test_split(df4, test_size=0.2, random_state=42)

#independent varibales
X_train2=df_train[['latitude scl','longitude scl']]
X_test2=df_test[['latitude scl','longitude scl']]


# Target for classification model
yC_train=df_train['classification'].ravel()
yC_test=df_test['classification'].ravel()
# Target for regression model
yR_train=df_train['classification'].ravel()
yR_test=df_test['classification'].ravel()


#---------- Step 3a - Set model parameters - Classification
modelC = KNeighborsClassifier(n_neighbors=5, #default=5
                              weights='uniform', #{‘uniform’, ‘distance’} or callable, default='uniform'
                              algorithm='auto', #{‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, default=’auto’
                              #leaf_size=30, #default=30, Leaf size passed to BallTree or KDTree.
                              #p=2, #default=2, Power parameter for the Minkowski metric.
                              #metric='minkowski', #default=’minkowski’, with p=2 is equivalent to the standard Euclidean metric.
                              metric_params=None, #dict, default=None, Additional keyword arguments for the metric function.
                              n_jobs=-1 #default=None, The number of parallel jobs to run for neighbors search, -1 means using all processors
                            ) 

modelR = KNeighborsRegressor(n_neighbors=10, #default=5
                             weights='uniform', #{‘uniform’, ‘distance’} or callable, default='uniform'
                             algorithm='auto', #{‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, default=’auto’
                             #leaf_size=30, #default=30, Leaf size passed to BallTree or KDTree.
                             #p=2, #default=2, Power parameter for the Minkowski metric.
                             #metric='minkowski', #default=’minkowski’, with p=2 is equivalent to the standard Euclidean metric.
                             metric_params=None, #dict, default=None, Additional keyword arguments for the metric function.
                             n_jobs=-1 #default=None, The number of parallel jobs to run for neighbors search, -1 means using all processors
                            )


#---------- Step 4 - Fit the two models
clf = modelC.fit(X_train2, yC_train)
reg = modelR.fit(X_train2, yR_train)
reg2 = modelR.fit(X_train2, yR_train)


#---------- Step 5 - Predict class labels / target values
# Predict on training data
pred_labels_tr = modelC.predict(X_train2)
#pred_values_tr = modelR.predict(X_train)
pred_values_tr2 = modelR.predict(X_train2)

# Predict on a test data
pred_labels_te = modelC.predict(X_test2)
#pred_values_te = modelR.predict(X_test)
pred_values_te2 = modelR.predict(X_test2)


#---------- Step 6 - Print model results
# Basic info about the model
#print('---------------------------------------------------------')
#print('****************** KNN Classification ******************')    
#print('Classes: ', clf.classes_)
#print('Effective Metric: ', clf.effective_metric_)
#print('Effective Metric Params: ', clf.effective_metric_params_)
#print('No. of Samples Fit: ', clf.n_samples_fit_)
#print('Outputs 2D: ', clf.outputs_2d_)
#print('--------------------------------------------------------')
#print("")

#print('*************** Evaluation on Test Data ***************')
#scoreC_te = modelC.score(X_test2, yC_test)
#print('Accuracy Score: ', scoreC_te)
# Look at classification report to evaluate the model
#print(classification_report(yC_test, pred_labels_te))
#print('--------------------------------------------------------')
#print("")

#print('*************** Evaluation on Training Data ***************')
#scoreC_tr = modelC.score(X_train2, yC_train)
#print('Accuracy Score: ', scoreC_tr)
# Look at classification report to evaluate the model
#print(classification_report(yC_train, pred_labels_tr))
#print('---------------------------------------------------------')


# Basic info about the model
#print("")
#print('****************** KNN Regression ******************')    
#print('Effective Metric: ', reg2.effective_metric_)
#print('Effective Metric Params: ', reg2.effective_metric_params_)
#print('No. of Samples Fit: ', reg2.n_samples_fit_)
#print("")
#scoreR_te = modelR.score(X_test2, yR_test)
#print('Test Accuracy Score: ', scoreR_te)
#scoreR_tr = modelR.score(X_train2, yR_train)
#print('Training Accuracy Score: ', scoreR_tr)

#print('---------------------------------------------------------')

#assesses the final number and assigns a risk level
def assess(result,var):
    result[0] = result[0] - var
    print(result[0])
    if result[0] <= 0.4999999:
        print("low risk")
    elif result[0] >= 0.5 and result[0] <= 1.19999999:
        print("medium risk")
    elif result[0] >= 1.2:
        print("high risk")


#prompts user for input for latitude and longitude
lat_in = input("Enter latitude : ")
long_in = input ("Enter longitude :")


lat_in = float(lat_in)
long_in = float(long_in)


from math import sqrt
 
# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
    distance = 0.0
    x = np.array2string(row2)
    #print(x)
    x = x[1:-1]
    temp = []
    temp = x.split()
    #print(temp)
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
    distance = euclidean_distance([lat_in,long_in], row)
    #print(row)
    #print(distance)

#sees if nearest neighbor is more than 20 degrees away than user inputted location    
neighbors = get_neighbors(coord, [lat_in,long_in], 1)
for neighbor in neighbors:
    print(neighbor)
    if abs(lat_in - neighbor[0]) > 20 or abs(long_in - neighbor[1]) > 20:
        var = 0.5
    else:
        var = 0.0


#print( modelC.predict([[25.6,96]]) )
result =  modelR.predict([[lat_in,long_in]])
assess(result,var)


