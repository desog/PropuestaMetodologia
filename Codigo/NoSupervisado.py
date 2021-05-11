import numpy as np
from scipy.cluster.vq import kmeans
import pandas as pd
from sklearn.neighbors import NearestCentroid
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
path = "https://raw.githubusercontent.com/desog/PropuestaMetodologia/main/Recursos/DataSets/forestfires.csv"
dataframe = pd.read_csv(path)
area = dataframe["area"]
mes = dataframe["month"]
resultados = dataframe.groupby("month").sum()["area"]
resultados.sort_index(inplace=True)
res = resultados.to_frame()
X = np.array(list(zip(mes,area)))

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)
clf = NearestCentroid()
clf.fit(X,y_kmeans)
print("Centroides: ",kmeans.cluster_centers_)
print("predicción centroide más cercano: " ,clf.predict([[4,4]]))
plt.xlabel("Aerolineas")
plt.ylabel("Pasajeros")
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=30, cmap='rainbow')
plt.show()