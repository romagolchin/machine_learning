import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.preprocessing import scale

data = pd.read_csv('./data/wine.csv')
ans = []
# for j in range(4):
#     ans.append(open('./answers/ans' + str(j + 1) + '.txt', 'w'))
# ans2 = open('./answers/ans2.txt', 'w')
k_opt = 1
sc = 0
for i in range(1, 51):
    clf = KNeighborsClassifier(n_neighbors=i)
    fold = KFold(n=data.shape[0], n_folds=5, shuffle=True, random_state=42)
    data_arr = data.values
    accuracy = cross_val_score(estimator=clf, X=scale(data_arr[0::, 1::]), y=data_arr[0::, 0], scoring='accuracy', cv=fold)
    if sc < accuracy.mean():
        sc = accuracy.mean()
        k_opt = i

print(k_opt)
print(sc)
# for j in range(4):
#     ans[j].close()
