import numpy as np
from utils import get_directory

FILE_ENCODING = [1, 0, 0]
PROCESS_ENCODING = [0, 1, 0]
SOCKET_ECODING = [0, 0, 1]

CMD_LINE_SIZE = 12
LOGIN_NAME_SIZE = 6
EUID_SIZE = 5
BINARY_FILE_SIZE = 4

EMPTY_CMD_LINE = [0] * CMD_LINE_SIZE
EMPTY_LOGIN_NAME = [0] * LOGIN_NAME_SIZE
EMPTY_EUID = [0] * EUID_SIZE
EMPTY_BINARY_FILE = [0] * BINARY_FILE_SIZE

CMD_LINE_CHOICES = range(0, CMD_LINE_SIZE)
LOGIN_NAME_CHOICES = range(0, LOGIN_NAME_SIZE)
EUID_CHOICES = range(0, EUID_SIZE)
BINARY_FILE_CHOICES = range(0, BINARY_FILE_SIZE)

CMD_LINE = [
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
]

ATTR_DIM = 30

DUMMY = [-1] * 30

BIG_NUMBER = 1000000000

EMPTY_TIMES_DICT = {
    "neigh_assembly": list(),
    "normalized_subgraph": list(),
    "canonicalizes": list(),
    "compute_subgraph_ranking": list(),
    "labeling_procedure": list(),
    "first_labeling_procedure": list()
}

KNN_GRID = {
    "neighbours": range(1, 30),
    "p_dist": range(1, 5)
}

RF_GRID = {
    "depth": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    "estimators": [10, 20, 40, 80, 160],
    "samples_split": [2, 5, 10],
    "samples_leaf": [1, 2, 4]
}

LOG_REG_GRID = {
    "c": np.logspace(-4, 4, 20),
    "penalty": ['l1', 'l2'],
    "width": [11, 13, 15, 17, 19],
    "stride": [1, 3, 5, 7, 9, 11],
    "rf_size": [2]
}

MLP_GRID = {
    "hidden_size": [32, 64, 128, 256, 512],
    "batch_size": [16, 32, 64, 128, 256],
    "epochs": [10, 30, 50, 70, 100],
    "learning_rate": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
    "dropout_rate": [0.1, 0.3, 0.5, 0.7, 0.9],
    "init_mode": ['uniform', 'normal', 'zero', 'he_normal', 'he_uniform']
}

CNN_GRID = {
    "width": [11, 13, 15, 17, 19],
    "stride": [1, 3, 5, 7, 9, 11],
    "rf_size": [2],
    # "hidden_size": [32, 64, 128, 256, 512],
    "batch_size": [16, 32, 64, 128, 256],
    "epochs": [10],
    "learning_rate": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
    "dropout_rate": [0.1, 0.3, 0.5, 0.7, 0.9],
    "init_mode": ['uniform', 'normal', 'zero', 'he_normal', 'he_uniform']
}

GRIDS = {
    "CNN": CNN_GRID,
    "MLP": MLP_GRID,
    "LRG": LOG_REG_GRID,
    "KNN": KNN_GRID,
    "RF": RF_GRID
}

LOG_DIRS = [
    get_directory() + '/Data_Sets/Logs/Android',
    get_directory() + '/Data_Sets/Logs/Apache',
    get_directory() + '/Data_Sets/Logs/Hadoop',
    get_directory() + '/Data_Sets/Logs/OpenStack',
    get_directory() + '/Data_Sets/Logs/Spark',
    get_directory() + '/Data_Sets/Logs/SSH'
]
