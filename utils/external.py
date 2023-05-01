import os
from argparse import Namespace
from datetime import datetime, timedelta

import h5py
import numpy as np

import pandas as pd


def parse_config(config, **kwargs):
    args = Namespace()
    for k, v in config.items():
        setattr(args, k, v)
    for k, v in kwargs.items():
        setattr(args, k, v)
    return args


def load_file(file_path):
    """
    Load CSV file with gzip compression
    :param file_path: path of the file
    :return: Dataframe
    """
    return pd.read_csv(file_path, compression='gzip')


def load_h5py(path):
    f = h5py.File(path, 'r')
    dt = pd.to_datetime(np.array(f['#/datetime64[ns]'])[0]).tz_localize('UTC').tz_convert('Asia/Shanghai')
    dt = dt.strftime('%Y-%m-%d %H:%M:%S')
    return pd.DataFrame({'DateTime': dt, 'Uid': np.array(f['#/object'])[0].astype(str),
                         'y': np.array(f['#/float64'])[0]})


def iter_files(data_dir):
    return [os.path.join(r, f[0]) for r, d, f in os.walk(data_dir) if len(f) != 0]


def get_trade_dates(data_dir, start_date='1990-01-01', end_date='2024-01-01'):
    file_paths = iter_files(data_dir)
    trade_dates = ['-'.join(file_path.split(os.path.sep)[-4:-1]) for file_path in file_paths]
    return [trade_date for trade_date in trade_dates if end_date > trade_date >= start_date]


def load_feat_uid(base_dir, folders):
    index_path = os.path.join(base_dir, 'indexes.csv')
    with open(index_path, 'r') as f:
        features_uid = [line.split(',') for line in f.readlines()]
    uid = features_uid[-1]
    feature = [features_uid[i] for i in range(5) if 'data' + str(i) in folders]
    return sorted(list(set().union(*feature))), uid


def _valid_type(d_type):
    return d_type in ['float64', 'float32', 'int64', 'int32']


def get_snap_time(yy, mm, dd):
    start_time = datetime(yy, mm, dd, 9, 30)
    return [(start_time + timedelta(minutes=5 * t)).strftime('%Y-%m-%d %H:%M:%S') for t in range(0, 7, 1)]
