import pandas as pd
import numpy as np


def first_name(str):
    str1 = "Miss."
    str2 = "."
    if str.find(str1) >= 0:
        return str[str.find(str1) + 6: str.find(" ", str.find('.'))]

data = pd.read_csv('train.csv', index_col='PassengerId')
women = data.loc[data.Sex == "female"]
men = data.loc[data.Sex == "male"]
print(women.shape[0])
print(men.shape[0])
survived = data.loc[data.Survived == 1]
firstclass = data.loc[data.Pclass == 1]
print(survived.shape[0] / data.shape[0])
print(firstclass.shape[0] / data.shape[0])
print(data.Age.mean())
print(data.Age.dropna().median())
print(women)
# notnull = data.loc[(not data.Parch.isnull()) & (not data.SibSp.isnull())]
print(data.corr('pearson'))
women_names = women.Name.values
print(map(first_name(str), women_names))
