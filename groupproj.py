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


df = pd.read_csv (r'D:/Downloads/GLC03122015_stripped.csv')

lat = df["latitude"]
lon = df["longitude"]

size = df["landslide1"]
fat = df["fatalities"]

lat = lat.to_numpy()
lon = lon.to_numpy()

test = [23.2,79.3]

coord = np.column_stack((lon, lat))
#coord = (lon, lat)

nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(coord)
print(nbrs)
distances, indices = nbrs.kneighbors(coord)

#print (distances)
#print (indices)

#after this point it doesn't work



landslide1 = df["landslide1"]
landslide1 = landslide1.to_numpy()
size_array = []
fat_array = []
for word in landslide1:
   # print(word)
    if word == 'Small':
        word = 0
        size_array.append(word)
    elif word == 'Medium':
        word = 2
        size_array.append(word)
    elif word == 'Large':
        word = 4
        size_array.append(word)
    elif word == 'Very_large':
        word = 9
        size_array.append(word)
    elif word == 'unknown':
        word = 0
        size_array.append(word)
    else:
        word = 0
        size_array.append(word)

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
        word = 7
        fat_array.append(word)
    elif count > 31:
        word = 9
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

K = 10
model = KNeighborsClassifier(n_neighbors = K)
#model.fit(coord, classification)
print(model)
print()

#print( '(32,77) is class'),
#print( model.predict([[32,77]]) )



# Do Min-Max scaling
scaler = MinMaxScaler()
df['fatalities scl']=scaler.fit_transform(df[['fatalities']])
df['landslide1 scl']=scaler.fit_transform(df[['landslide1']])
df['latitude scl']=scaler.fit_transform(df[['latitude']])
df['longitude scl']=scaler.fit_transform(df[['longitude']])

# Split into train and test dataframes
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

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
yR_train=df_train['class'].ravel()
yR_test=df_test['class'].ravel()



modelR = KNeighborsRegressor(n_neighbors=3, #default=5
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



print( '(25,96) is class'),
print( modelR.predict([[25.6,96]]) )

print( '(-43.5,170) is class'),
print( modelR.predict([[-43.5,170]]) )