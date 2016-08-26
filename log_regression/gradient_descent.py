import pandas
import numpy
from sklearn.metrics import roc_auc_score


# import math


def norm(a, b):
    return (a ** 2 + b ** 2) ** 0.5


def sigma(x):
    return 1 / (1 + numpy.exp(-x))


data = pandas.read_csv('./data/data-logistic.csv')
w = [0, 0]
it = 0
d0, d1 = 1, 1
l = data.shape[0]
k = 0.1
C = 0
y, x0, x1 = data.values[0::, 0], data.values[0::, 1], data.values[0::, 2]
while (it < 10000) and (norm(d0, d1) > 1e-5):
    res0, res1 = 0, 0
    w_old = w[:]
    for i in range(l):
        M = y[i] * (x0[i] * w[0] + x1[i] * w[1])
        res0 += y[i] * x0[i] * (1 - sigma(M))
        res1 += y[i] * x1[i] * (1 - sigma(M))
    w[0] += k * (res0 / l - C * w[0])
    w[1] += k * (res1 / l - C * w[1])
    d0, d1 = w[0] - w_old[0], w[1] - w_old[1]
    it += 1
print(w)
predict = []
for i in range(l):
    M = y[i] * (x0[i] * w[0] + x1[i] * w[1])
    predict.append(sigma(M))
print(numpy.asarray(predict))
score = roc_auc_score(y, numpy.asarray(predict))
print(score)
