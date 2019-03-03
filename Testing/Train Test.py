from data_loader import load_local_data
from ML_Module.CNN import PSCN
from sklearn.model_selection import train_test_split
import numpy as np

mutag_dataset = load_local_data('/home/tiberiu/PycharmProjects/Part2Project/data', 'cox2', attributes=True)
X, y = zip(*mutag_dataset)

pscn = PSCN(w=18, k=10, attr_dim=3, epochs=100, batch_size=32,
            verbose=2, gpu=True)  # see receptive_field_maker_example for more details

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

pscn.fit(X_train, y_train)

preds = pscn.predict(X_test)

print(np.sum(preds.ravel() == y_test) / len(y_test))
