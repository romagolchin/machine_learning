import csv as csv 
import numpy as np

train_file = csv.reader(open('train.csv', 'rb')) 
header = train_file.next()

data = []                          
for row in train_file:      
    data.append(row)             
data = np.array(data) 	         

num_classes = 3
num_bins = 4
survived_table = np.zeros((2, num_classes, num_bins))
bracket_size = 10
ceil = bracket_size * num_bins

for i in xrange(2):
	for j in xrange(3):
		for k in xrange(4):
			if i == 0:
				gender = "female"
			else:
				gender = "male"
			cur_surv = data[ \
						(data[0::,4] == gender) & \
						(data[0::,2].astype(np.float) == j + 1) & \
						(data[0:, 9].astype(np.float) >= k * bracket_size) & \
						(data[0:, 9].astype(np.float) < (k + 1) * bracket_size) \
						,1]
			if np.size(cur_surv) != 0:
				survived_table[i, j, k] = np.mean(cur_surv.astype(np.float))
			else:
				survived_table[i, j, k] = 0.0
print survived_table

survived_table[survived_table >= 0.5] = 1
survived_table[survived_table < 0.5] = 0
print survived_table

test_file = open('test.csv', 'rb')
test_file_object = csv.reader(test_file)
header = test_file.next()

prediction3 = open('prediction3.csv', 'wb')
prediction3_object = csv.writer(prediction3)

prediction3_object.writerow(["PassengerId", "Survived"])
q = 0
for row in test_file_object:
	for j in xrange(num_bins):
		# print row[0]
		try:
			row[8] = float(row[8])
		except:
			bin = 3 - float(row[1])
			break
		if row[8] >= ceil:
			bin = num_bins - 1.0
			break
		if (row[8] >= j * bracket_size) & (row[8] < (j + 1) * bracket_size):
			bin = j
			break
		# print bin
	if row[3] == "female":
		prediction3_object.writerow([row[0], "%d" % int (survived_table[0, float(row[1]) - 1, bin] ) ])  
	else:
		prediction3_object.writerow([row[0], "%d" % int (survived_table[1, float(row[1]) - 1, bin] ) ])
	q += 1
print q
test_file.close()
prediction3.close()