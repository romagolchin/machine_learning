import csv as csv 
import numpy as np
import pandas as pd

csv_file_object = csv.reader(open('train.csv', 'rb')) 
header = csv_file_object.next() 

data=[]                          
for row in csv_file_object:      
    data.append(row)             
data = np.array(data) 	         

women_row = data[0::,4] == "female"
men_row = data[0::,4] == "male"
women = data[women_row, 1].astype(np.float)
men = data[men_row, 1].astype(np.float)
prop_women_surv = np.sum(women) / np.size(women)
prop_men_surv = np.sum(men) / np.size(men)
print(np.size(women))
print(np.size(men))
print(prop_women_surv)
print(prop_men_surv)
