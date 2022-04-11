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


df = pd.read_csv (r'D:/Downloads/Global_Landslide_Catalog_Export_stripped.csv')

lat = df["latitude"]
lon = df["longitude"]

size = df["landslide_size"]
fat = df["fatality_count"]

lat = lat.to_numpy()
lon = lon.to_numpy()

test = [23.2,79.3]

coord = np.column_stack((lon, lat))
#coord = (lon, lat)

nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(coord)
print(nbrs)
distances, indices = nbrs.kneighbors(coord)

#print (distances)
#print (indices)

#after this point it doesn't work



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

size_array2 = []
for word in landslide1:
   # print(word)
    if word == 0:
        word = 0
        size_array2.append(word)
    elif word == 1:
        word = 2
        size_array2.append(word)
    elif word == 2:
        word = 4
        size_array2.append(word)
    elif word == 3:
        word = 7
        size_array2.append(word)
    else:
        word = 0
        size_array2.append(word)

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

i = 0
class_array =[]
for i in range(len(fat_array)):
    temp = fat_array[i] + size_array[i]
    class_array.append(temp)



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



#print(size_array)
#print(fat_array)
#print(class_array)
#print(classification)
#print(coord)


from csv import writer
from csv import reader

#print(len(classification))
classification2 = classification
classification2.insert(0, "classification")


# Open the input_file in read mode and output_file in write mode
with open('D:/Downloads/Global_Landslide_Catalog_Export_stripped.csv', encoding="utf8") as read_obj, \
        open('D:/Downloads/Global_Landslide_Catalog_Export2.csv', 'w', newline='') as write_obj:
    # Create a csv.reader object from the input file object
    csv_reader = reader(read_obj)
    # Create a csv.writer object from the output file object
    csv_writer = writer(write_obj)
    # Read each row of the input csv file as list
    count = 0
    for row in csv_reader:
        #print(count)
        # Append the default text in the row / list
        #print(classification[count])
        temp = classification[count]
        row.append(temp)
        count = count + 1
        # Add the updated row / list to the output file
        csv_writer.writerow(row)


K = 10
model = KNeighborsClassifier(n_neighbors = K)
#model.fit(coord, classification)
print(model)
print()

#print( '(32,77) is class'),
#print( model.predict([[32,77]]) )

df2 = pd.read_csv (r'D:/Downloads/Global_Landslide_Catalog_Export2.csv')


# Do Min-Max scaling
scaler = MinMaxScaler()
#df2['fatalities scl']=scaler.fit_transform(df2[['fatality_count']])
#df2['landslide1 scl']=scaler.fit_transform(df2[['landslide_size']])
df2['latitude scl']=scaler.fit_transform(df2[['latitude']])
df2['longitude scl']=scaler.fit_transform(df2[['longitude']])

# Split into train and test dataframes
df_train, df_test = train_test_split(df2, test_size=0.2, random_state=42)

# Independent variables (features)
#X_train=df_train[['fatalities scl', 'landslide1 scl', 'latitude scl','longitude scl']]
#X_test=df_test[['fatalities scl', 'landslide1 scl', 'latitude scl','longitude scl']]

X_train2=df_train[['latitude','longitude']]
X_test2=df_test[['latitude','longitude']]

#print(X_test)
print(X_test2)
# Target for classification model
#yC_train=df_train['Price Band enc'].ravel()
#yC_test=df_test['Price Band enc'].ravel()
# Target for regression model
yR_train=df_train['classification'].ravel()
yR_test=df_test['classification'].ravel()



modelR = KNeighborsRegressor(n_neighbors=4, #default=5
                             weights='uniform', #{‘uniform’, ‘distance’} or callable, default='uniform'
                             algorithm='auto', #{‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, default=’auto’
                             #leaf_size=30, #default=30, Leaf size passed to BallTree or KDTree.
                             #p=2, #default=2, Power parameter for the Minkowski metric.
                             #metric='minkowski', #default=’minkowski’, with p=2 is equivalent to the standard Euclidean metric.
                             metric_params=None, #dict, default=None, Additional keyword arguments for the metric function.
                             n_jobs=-1 #default=None, The number of parallel jobs to run for neighbors search, -1 means using all processors
                            )


#---------- Step 4 - Fit the two models
#clf = modelC.fit(X_train, yC_train)
#reg = modelR.fit(X_train, yR_train)
reg2 = modelR.fit(X_train2, yR_train)


#---------- Step 5 - Predict class labels / target values
# Predict on training data
#pred_labels_tr = modelC.predict(X_train)
#pred_values_tr = modelR.predict(X_train)
pred_values_tr2 = modelR.predict(X_train2)
#print(pred_values_tr)
print(pred_values_tr2)

# Predict on a test data
#pred_labels_te = modelC.predict(X_test)
#pred_values_te = modelR.predict(X_test)
#print(pred_values_te)
pred_values_te2 = modelR.predict(X_test2)
print(pred_values_te2)


##pred_values_te2 = modelR.predict([['fatalities scl', 'landslide1 scl', 'latitude scl','longitude scl']])
#print(pred_values_te2)





# Basic info about the model
print("")
print('****************** KNN Regression ******************')    
#print('Effective Metric: ', reg.effective_metric_)
#print('Effective Metric Params: ', reg.effective_metric_params_)
#print('No. of Samples Fit: ', reg.n_samples_fit_)
print("")
#scoreR_te = modelR.score(X_test, yR_test)
#print('Test Accuracy Score: ', scoreR_te)
#scoreR_tr = modelR.score(X_train, yR_train)
#print('Training Accuracy Score: ', scoreR_tr)

print('---------------------------------------------------------')


# Basic info about the model
print("")
print('****************** KNN Regression ******************')    
print('Effective Metric: ', reg2.effective_metric_)
print('Effective Metric Params: ', reg2.effective_metric_params_)
print('No. of Samples Fit: ', reg2.n_samples_fit_)
print("")
scoreR_te = modelR.score(X_test2, yR_test)
print('Test Accuracy Score: ', scoreR_te)
scoreR_tr = modelR.score(X_train2, yR_train)
print('Training Accuracy Score: ', scoreR_tr)

print('---------------------------------------------------------')



def assess(result):
    print(result[0])
    if result[0] <= 0.4999999:
        print("low risk")
    elif result[0] >= 0.5 and result[0] <= 1.19999999:
        print("medium risk")
    elif result[0] >= 1.2:
        print("high risk")



lat_in = input("Enter latitude : ")
long_in = input ("Enter longitude :")


print("(",lat_in,",",long_in,") is a "),
#print( modelR.predict([[25.6,96]]) )
result =  modelR.predict([[lat_in,long_in]])
assess(result)

lat_in = float(lat_in)
long_in = float(long_in)

# Example of getting neighbors for an instance
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

    
neighbors = get_neighbors(coord, [lat_in,long_in], 1)
for neighbor in neighbors:
	print(neighbor)


