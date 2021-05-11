import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error,r2_score
#Carga de datos
path = "https://raw.githubusercontent.com/desog/PropuestaMetodologia/main/Recursos/DataSets/space-shuttle/o-ring-erosion-only.csv"
dataframe = pd.read_csv(path)
rasgos_prediccion = ["Experiencing thermal distress","Launch temperature (degrees F)","Leak-check pressure (psi)"]
x = dataframe[rasgos_prediccion]
y = dataframe["Experiencing thermal distress"]
#Elección de variables predictoras
estimator = SVR(kernel = "linear")
selector = RFE(estimator,n_features_to_select=2,step=1)
selector = selector.fit(x,y)
x_pred = x[["Launch temperature (degrees F)"]]
#Selección de grado de la regresión
poly_degree =2
polynomial_features = PolynomialFeatures(degree = poly_degree)
x_transform = polynomial_features.fit_transform(x_pred)
model = LinearRegression().fit(x_transform, y)
y_new = model.predict(x_transform)
#  rmse y r2
rmse = np.sqrt(mean_squared_error(y, y_new))
r2 = r2_score(y, y_new)
print('RMSE: ', rmse)
print('R2: ', r2)
# prediccion
x_new_min = 0.0
x_new_max = 70
x_new = np.linspace(x_new_min, x_new_max, 70)
x_new = x_new[:,np.newaxis]
x_new_transform = polynomial_features.fit_transform(x_new)
y_new = model.predict(x_new_transform)

#Grafico
plt.plot(x_new, y_new, color='coral', linewidth=3)
plt.grid()
plt.xlim(x_new_min,x_new_max)
plt.ylim(0,100)

plt.title("Regresión lineal \n ", fontsize=10)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
