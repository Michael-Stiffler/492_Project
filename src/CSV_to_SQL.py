
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
import os

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
import pyodbc
from sqlite3 import Cursor


def connect_sql():
    #Connect to SQL server
    database = "WILLS_DB"
    server = 'DESKTOP-TUJPIMN'
    connect = pyodbc.connect("DRIVER={SQL Server};SERVER="+server+";Trusted_Connection=yes;")
    connect.autocommit = True
    cursor = connect.cursor()

    #Create database 
    createdb = "IF DB_ID('"+database+"') IS NULL CREATE DATABASE "+database+";"
    cursor.execute(createdb)

    #Switches to new database
    switchdb = "USE "+database+";"
    cursor.execute(switchdb)

    #tableinfo = "CREATE TABLE dbo.landslide(landslide_size VARCHAR(14),fatality_count VARCHAR(10),date VARCHAR(15),country_name VARCHAR(30),cause VARCHAR(30),injuries VARCHAR(30),type VARCHAR(30),longitude float,latitude float,classification int);"
    #cursor.execute(tableinfo)
    return cursor


#connect to SQL server and create database/table
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
    #loc = input("Enter stripped data location:")
    import pathlib
    path_parent = os.path.dirname(pathlib.Path(__file__).parent.resolve())
    os.chdir(path_parent)
    path = os.getcwd()
    path = path + "/data/Global_Landslide_Catalog_Export_stripped.csv"
    loc = path.replace("/", "\\")
    loc2 = loc.replace(".csv","2.csv")
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
    print("Done!")

