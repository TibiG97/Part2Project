from data_loader import load_local_data
from Evaluation_Module.Nested_Cross_Validation import nested_cross_validation

import sys

mutag_dataset = load_local_data('/home/tiberiu/PycharmProjects/Part2Project', 'node2')
X, y = zip(*mutag_dataset)

nested_cross_validation(X, y, 10, 1)

sys.exit()
