import numpy as np
from scipy.cluster.vq import kmeans

X = np.array([[3.0,2.0],[2.0,2.0],[2.0,1.0],[1.0,2.0],[3.0,6.0],[7.0,4.0],
            [8.0,4.0],[8.0,3.0],[9.0,4.0],[9.0,5.0]])
Kmeans = kmeans(X,3)
print(Kmeans) #Centroides
