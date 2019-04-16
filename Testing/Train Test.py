from data_loader import load_local_data
from ML_Module.CNN import CNN
from sklearn.model_selection import train_test_split
import numpy as np
from Evaluation_Module.Nested_Cross_Validation import nested_cross_validation

import sys

mutag_dataset = load_local_data('/home/tiberiu/PycharmProjects/Part2Project', 'mine')
X, y = zip(*mutag_dataset)

nested_cross_validation(X, y, 10, 10)

sys.exit()
