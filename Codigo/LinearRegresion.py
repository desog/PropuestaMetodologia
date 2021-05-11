import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
path = "https://raw.githubusercontent.com/desog/PropuestaMetodologia/main/Recursos/DataSets/Computer%20Hardware/machinedata.csv"
dataframe = pd.read_csv(path)
train, test = train_test_split(dataframe, test_size = 0.9, random_state = 42)
rasgos_prediccion = ["MYCT","MMIN","MMAX"]
x = train[rasgos_prediccion]
y = train["PRP"]
print(x.shape)
estimator = SVR(kernel = "linear")
selector = RFE(estimator,n_features_to_select=2,step=1)
selector = selector.fit(x,y)
print(selector.support_)