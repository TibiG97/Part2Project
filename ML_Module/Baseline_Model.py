from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np

cls = DecisionTreeClassifier()

X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=2, random_state=1,
                           n_clusters_per_class=1)
f = open('/home/tiberiu/PycharmProjects/Part2Project/Data_Processing/node_attributes.txt', 'r')

array = [[int(x[0]) for x in line.split()] for line in f]

size = len(array)

array = [array[i] + (array[i + 1] if i + 1 < size else []) for i in range(0, size, 2)]
array = np.array(array)

X = array
y = list()
for iterator in range(0, 200):
    if iterator < 100:
        y.append(1)
    else:
        y.append(-1)

y = np.array(y)

classifiers = [DecisionTreeClassifier(max_depth=5),
               RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
               KNeighborsClassifier(n_neighbors=3)]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

print(X_train, y_train)

for classifier in classifiers:
    classifier.fit(X_train, y_train)
    print(classifier.score(X_test, y_test))
