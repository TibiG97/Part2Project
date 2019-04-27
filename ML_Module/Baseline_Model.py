from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np

cls = DecisionTreeClassifier()

f = open('/home/tiberiu/PycharmProjects/Part2Project/Data_Processing/node_attributes.txt', 'r')

array = [[int(x[0]) for x in line.split()] for line in f]

size = len(array)

array = [array[i] + (array[i + 1] if i + 1 < size else []) for i in range(0, size, 2)]

'''
aux_array = []
for iterator in range(0, size, 10):
    feature_vector = []
    for index in range(0, 10):
        feature_vector += array[index]
    aux_array.append(feature_vector)

array = np.array(aux_array)
'''
print(array)
print(len(array))
print(len(array[0]))

X = array
y = list()
for iterator in range(0, 2000):
    if iterator < 500:
        y.append(1)
    elif iterator < 1000:
        y.append(2)
    elif iterator < 1500:
        y.append(3)
    else:
        y.append(4)

y = np.array(y)

classifiers = [DecisionTreeClassifier(max_depth=5),
               RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
               KNeighborsClassifier(n_neighbors=3)]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=42)

print(X_train, y_train)

for classifier in classifiers:
    classifier.fit(X_train, y_train)
    print(y_test)
    print(classifier.predict(X_test))
    print(classifier.score(X_test, y_test))

