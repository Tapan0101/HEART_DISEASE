# This is a sample Python script.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle

# read data
df = pd.read_excel('Training_data (1).xlsx')
print(df.head())

## Split data



y = df['Electrical Power (EP)']
X = df.iloc[:, :4]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=101)

# scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_val)

## Model

lm = LinearRegression()
lm.fit(X_train,y_train)

pickle.dump(lm,open("model.pkl","wb"))

import sys
print("Python version")
print(sys.version)
print("Version info.")
print(sys.version_info)

