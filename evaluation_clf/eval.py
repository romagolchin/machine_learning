import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve

clf = pd.read_csv('./data/classification.csv')
f1 = open('./answers/ans1.txt', 'w')
f2 = open('./answers/ans2.txt', 'w')
f3 = open('./answers/ans3.txt', 'w')
f4 = open('./answers/ans4.txt', 'w')
tp = clf.loc[(clf.true == 1) & (clf.pred == 1)]
fp = clf.loc[(clf.true == 0) & (clf.pred == 1)]
tn = clf.loc[(clf.true == 0) & (clf.pred == 0)]
fn = clf.loc[(clf.true == 1) & (clf.pred == 0)]
print(tp.shape[0], fp.shape[0], fn.shape[0], tn.shape[0], sep=' ', file=f1)
acc = (tp.shape[0] + tn.shape[0]) / clf.shape[0]
p = tp.shape[0] / (tp.shape[0] + fp.shape[0])
r = tp.shape[0] / (tp.shape[0] + fn.shape[0])
fm = 2 * p * r / (p + r)
print(acc, p, r, fm, sep=' ', file=f2)
scores = pd.read_csv('./data/scores.csv').values
pr = 0
for i in range(4):
    print(roc_auc_score(scores[0::, 0], scores[0::, 1 + i]))
    prec, rec, thr = precision_recall_curve(scores[0::, 0], scores[0::, 1 + i])
    l = len(rec) - 1
    for j in range(l + 1):
        if (rec[j] >= 0.7) and (prec[j] > pr):
            pr = prec[j]
            ind = i
print(ind, pr)
# print(sep=' ', file=f4)