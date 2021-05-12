from sklearn import preprocessing
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
path = "https://raw.githubusercontent.com/desog/PropuestaMetodologia/main/Recursos/DataSets/Car%20Evaluation/car.csv"
dataframe = pd.read_csv(path)
print(dataframe.head())
train, test = train_test_split(dataframe, test_size = 0.2)
#Obteniendo variables 
buying = train["buying"]
maint = train["maint"]
doors = train["doors"]
persons = train["persons"]
lug_boot = train["lug_boot"]
safety = train["safety"]
clases = train["class"]
# Transformando variables 
le = preprocessing.LabelEncoder()
buying_encoded = le.fit_transform(buying)
maint_encoded = le.fit_transform(maint)
doors_encoded = le.fit_transform(doors)
persons_encoded = le.fit_transform(persons)
lug_boot_encoded = le.fit_transform(lug_boot)
safety_encoded = le.fit_transform(safety)
clases_encoded = le.fit_transform(clases)
combinacion = list(zip(buying_encoded,maint_encoded,doors_encoded,persons_encoded,lug_boot_encoded,safety_encoded))
print(clases)
print(clases_encoded)
#Modelado
model = MLPClassifier(activation='logistic',max_iter=700,hidden_layer_sizes=(4,),alpha=0.001,solver='lbfgs')
model.fit(combinacion,clases_encoded)
#Test
buyingtest = test["buying"]
mainttest = test["maint"]
doorstest = test["doors"]
personstest = test["persons"]
lug_boottest = test["lug_boot"]
safetytest = test["safety"]
clasestest = test["class"]
le = preprocessing.LabelEncoder()
buyingtest_encoded = le.fit_transform(buyingtest)
mainttest_encoded = le.fit_transform(mainttest)
doorstest_encoded = le.fit_transform(doorstest)
personstest_encoded = le.fit_transform(personstest)
lug_boottest_encoded = le.fit_transform(lug_boottest)
safetytest_encoded = le.fit_transform(safetytest)
clasestest_encoded = le.fit_transform(clasestest)
combinaciontest = list(zip(buyingtest_encoded,mainttest_encoded,doorstest_encoded,personstest_encoded,lug_boottest_encoded,safetytest_encoded))
print(clasestest)
print(clasestest_encoded)
#Predicción
print('Predicciones: ',model.predict(combinaciontest))

