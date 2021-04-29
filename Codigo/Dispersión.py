import matplotlib.pyplot as plt
import numpy as np
dataSet = np.array([[3,2],[2,2],[2,1],[1,2],[9,5],[9,4],[7,4],[8,3],
[4,3],[8,4],[1,1],[1,3],[3,3],[8,4.5],[8,5]])
plt.scatter(dataSet[:,0],dataSet[:,1],label = ' True Position')
plt.show()

