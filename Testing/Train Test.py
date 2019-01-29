from data_loader import load_local_data
import networkx as nx
from pscn import PSCN
from sklearn.model_selection import train_test_split
import numpy as np

mutag_dataset = load_local_data('./data', 'mutag')
X, y = zip(*mutag_dataset)

pscn = PSCN(w=18, k=10, one_hot=29, epochs=100, batch_size=32,
            verbose=2, gpu=1)  # see receptive_field_maker_example for more details

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

pscn.fit(X_train, y_train)

preds = pscn.predict(X_test)

print(np.sum(preds.ravel() == y_test) / len(y_test))
