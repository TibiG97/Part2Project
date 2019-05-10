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

CMD_LINE_DICT = {
    0: 'gunzip',
    1: 'ssh',
    2: 'rm',
    3: 'dd',
    4: 'sleep',
    5: 'postgres',
    6: 'cat',
    7: 'bounce',
    8: 'uname',
    9: 'gzip',
    10: 'mv',
    11: 'unlink'
}

LOGIN_NAME_DICT = {
    0: 'steve',
    1: 'operator',
    2: 'lariat',
    3: 'darpa',
    4: 'root',
    5: 'none'
}

CLASS_DICT = {
    1: 'android',
    2: 'apache',
    3: 'hadoop',
    4: 'openstack',
    5: 'spark',
    6: 'ssh'
}

ATTR_DIM = 30

DUMMY = [-1] * 30

BIG_NUMBER = 1000000000

KNN_GRID = {
    "neighbours": range(400, 501),
    "p_dist": range(2, 3)
}

RF_GRID = {
    "depth": [50, 60, 70, 80, 90, 100],
    "estimators": [200, 250, 300, 350],
    "samples_split": [2, 5, 10],
    "samples_leaf": [1, 2, 4]
}

MLP_GRID = {
    "hidden_size": [32, 64, 128, 256, 512],
    "batch_size": [16, 32, 64, 128, 256],
    "epochs": [10, 30, 50, 70, 100],
    "learning_rate": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
    "dropout_rate": [0.1, 0.3, 0.5, 0.7, 0.9]
}

CNN_GRID = {
    "width": [13, 14, 15, 16, 17],
    "stride": [1],
    "rf_size": [4, 5, 6],
    # "hidden_size": [32, 64, 128, 256, 512],
    "batch_size": [32, 64, 128],
    "epochs": [50, 100, 200],
    "learning_rate": [0.001, 0.005],
    "dropout_rate": [0.1, 0.3, 0.5]
}

GRIDS = {
    "CNN": CNN_GRID,
    "MLP": MLP_GRID,
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
