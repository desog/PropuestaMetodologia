import matplotlib.pyplot as plt
import numpy as np
x = np.array([2,3,6,8,10,12,14,16,20,26,28,30,34,36,38,44])
y = range(16)
plt.plot(x, y, color='blue', linewidth=3)
plt.title("Gráfico de lineas ", fontsize=10)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

