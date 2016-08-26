import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.preprocessing import scale
from sklearn.datasets import load_boston

features = load_boston().data
target = load_boston().target
p_opt = 1.0
sc = -100.0
for cur_p in np.linspace(1.0, 10.0, 200):
    clf = KNeighborsRegressor(p=cur_p, weights='distance')
    fold = KFold(n=features.shape[0], n_folds=5, shuffle=True, random_state=42)
    # data_arr = data.values
    scores = cross_val_score(estimator=clf, X=scale(features), y=target, scoring='mean_squared_error', cv=fold)
    print(scores.mean())
    if sc < scores.mean():
        sc = scores.mean()
        p_opt = cur_p

print(p_opt)
print(-sc)
# for j in range(4):
#     ans[j].close()
