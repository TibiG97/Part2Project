from data_loader import load_local_data
from utils import get_directory
from Data_Processing.intermediar_data_loader import load_local_data_set
from Evaluation_Module.Nested_Cross_Validation import nested_cross_validation

import sys

# mutag_dataset = load_local_data(get_directory(), 'node2')
mutag_dataset = load_local_data_set('NEWSET', 'patchy_san')
X, y = mutag_dataset

old_dataset = load_local_data(get_directory(), 'node2')
xa, ya = zip(*old_dataset)

print(X[0].nodes())
print(X[0].values())
print(X[0].edges())

print(xa[0].nodes())
print(xa[0].edges())
print(xa[0].values())
sys.exit()
nested_cross_validation(X, y, 10, 1)

sys.exit()
