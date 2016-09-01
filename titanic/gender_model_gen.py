import csv as csv 
import numpy as np

test_file = open('test.csv', 'rb')
test_file_object = csv.reader(test_file)
header = test_file.next()

prediction2 = open('prediction2.csv', 'wb')
prediction2_object = csv.writer(prediction2)

prediction2_object.writerow(["PassengerId", "Survived"])
for row in test_file_object:
	if row[3] == "female":
		prediction2_object.writerow([row[0], '1'])
	else:
		prediction2_object.writerow([row[0], '0'])
test_file.close()
prediction2.close()