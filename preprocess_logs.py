from utils import get_directory
import pandas as pd
import glob
from ML_Module.mlp import MultilayerPerceptron


def read_data(logfile_path):
    log_collection = pd.DataFrame()
    logs = pd.DataFrame()
    logfiles = glob.glob(logfile_path + "/*.log")  # Get list of log files
    for logfile in logfiles:
        logs = pd.read_csv(logfile, sep="\n", header=None, names=['data'])
        logs['type'] = logfile.split('/')[-1]
        # Add log file data and type to log collection
        log_collection = log_collection.append(logs)

    # Remove empty lines
    log_collection = log_collection.dropna()
    # Reset the index
    log_collection = log_collection.reset_index(drop=True)

    return log_collection


import os

android_dir = get_directory() + '/DataSets/Logs/Android'
apache_dir = get_directory() + '/DataSets/Logs/Apache'
hadoop_dir = get_directory() + '/DataSets/Logs/Hadoop'
open_dir = get_directory() + '/DataSets/Logs/OpenStack'
spark_dir = get_directory() + '/DataSets/Logs/Spark'
ssh_dir = get_directory() + '/DataSets/Logs/SSH'

dirs = [android_dir,
        apache_dir,
        hadoop_dir,
        open_dir,
        spark_dir,
        ssh_dir]

log_collection = MultilayerPerceptron.read_data(apache_dir)
X, y = MultilayerPerceptron.prepare_data(log_collection['data'], log_collection['type'])
print(X.shape[1])
