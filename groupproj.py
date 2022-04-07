import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

df = pd.read_csv (r'D:/Downloads/GLC03122015_half.csv')

lat = df["latitude"]
lon = df["longitude"]
lat = lat.to_numpy()
lon = lon.to_numpy()

coord = np.column_stack((lon, lat))

nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(coord)
distances, indices = nbrs.kneighbors(coord)

#print (distances)
#print (indices)

#after this point it doesn't work



landslide1 = df["landslide1"]
landslide1 = landslide1.to_numpy()
landslide = []
for word in landslide1:
   # print(word)
    if word == 'Small':
        word = 0
        landslide.append(word)
    elif word == 'Medium':
        word = 1
        landslide.append(word)
    elif word == 'Large':
        word = 2
        landslide.append(word)
    elif word == 'Very_large':
        word = 3
        landslide.append(word)
    elif word == 'unknown':
        word = 4
        landslide.append(word)
    else:
        word = 5
        landslide.append(word)

print(landslide)

n_neighbors = 15

# import some data to play with
iris = datasets.load_iris()

# we only take the first two features. We could avoid this ugly
# slicing by using a two-dim dataset
X = coord
y = landslide


h = 0.02  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(["orange", "cyan", "cornflowerblue"])
cmap_bold = ["darkorange", "c", "darkblue"]

for weights in ["uniform", "distance"]:
    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X, y)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    sns.scatterplot(
        x=X[:, 0],
        y=X[:, 1],
        hue=iris.target_names[y],
        palette=cmap_bold,
        alpha=1.0,
        edgecolor="black",
    )
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(
        "3-Class classification (k = %i, weights = '%s')" % (n_neighbors, weights)
    )
    plt.xlabel("Latitude")
    plt.ylabel("Longitude")

plt.show()

