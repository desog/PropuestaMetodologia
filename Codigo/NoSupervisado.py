import numpy as np
from scipy.cluster.vq import kmeans
import pandas as pd
path = "https://raw.githubusercontent.com/desog/PropuestaMetodologia/main/Recursos/DataSets/incd.csv"
dataframe = pd.read_csv(path, encoding = 'latin1')