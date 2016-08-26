import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

train = pd.read_csv('./data/perceptron-train.csv')
test = pd.read_csv('./data/perceptron-test.csv')
X_train, y_train = train.values[0::, 1::], train.values[0::, 0]
X_test, y_test = test.values[0::, 1::], test.values[0::, 0]
perceptron = Perceptron(random_state=241)
perceptron = perceptron.fit(X_train, y_train)
prediction = perceptron.predict(X_test)
score = accuracy_score(y_test, prediction)
print(score)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
perceptron = perceptron.fit(X_train_scaled, y_train)
prediction = perceptron.predict(X_test_scaled)
score1 = accuracy_score(y_test, prediction)
print(score1 - score)