import csv
import warnings
from functools import reduce

from pandas import HDFStore
from tables import NaturalNameWarning

from core.torch_io import OpeningData
from utils.external import load_file, load_h5py, iter_files, get_trade_dates, load_feat_uid, _valid_type, \
    get_snap_time
import numpy as np
import time
import os
import pandas as pd


def compute_mean(base_dir, proc_dir):
    folders = ['data0', 'data1', 'data2', 'data3', 'data4']
    dataset = OpeningData(
        base_dir=base_dir,
        proc_dir=proc_dir,
        target_uid=None,
        folders=folders,
        period=('2018-01-01', '2019-07-01')
    )
    feat, uid = load_feat_uid(base_dir, folders)
    sum_mean = np.zeros(len(feat)), np.zeros(len(feat))
    num_data = len(dataset)
    t0 = time.time()
    counter = np.zeros(len(feat), dtype=np.int64)
    for i in range(num_data):
        x, y = dataset.__getitem__(i)
        invalid = np.any([x == 0, np.isnan(x), np.isinf(x)], axis=0)
        counter += np.sum(~invalid, axis=(0, 1))
        x[invalid] = 0
        sum_mean += np.sum(x, axis=(0, 1))
        print("Progress: {0:3d} / {1:3d}. Time Spent {2:.2f} Min".format(i, num_data, (time.time() - t0) / 60))
    mean = sum_mean / counter
    mean[counter == 0] = 0
    df = pd.DataFrame(mean, index=feat)
    df.T.to_csv(os.path.join(base_dir, 'mean.csv'))
    return


def compute_std(base_dir, proc_dir):
    folders = ['data0', 'data1', 'data2', 'data3', 'data4']
    dataset = OpeningData(
        base_dir=base_dir,
        proc_dir=proc_dir,
        target_uid=None,
        folders=folders,
        period=('2018-01-01', '2019-07-01')
    )
    feat, uid = load_feat_uid(base_dir, folders)
    mean = pd.read_csv(os.path.join(base_dir, 'mean.csv'), index_col=0).to_numpy()
    sum_std = np.zeros(len(feat))
    num_data = len(dataset)
    t0 = time.time()
    counter = np.zeros(len(feat), dtype=np.int64)
    for i in range(num_data):
        x, y = dataset.__getitem__(i)
        invalid = np.any([x == 0, np.isnan(x), np.isinf(x)], axis=0)
        counter += np.sum(~invalid, axis=(0, 1))
        square = np.power(x - mean, 2)
        square[invalid] = 0
        sum_std += np.sum(square, axis=(0, 1))
        print("Progress: {0:3d} / {1:3d}. Time Spent {2:.2f} Min".format(i, num_data, (time.time() - t0) / 60))
    std = sum_std / counter
    std[counter == 0] = 1
    std[std < 1e-4] = 1
    df = pd.DataFrame(np.sqrt(std), index=feat)
    df.T.to_csv(os.path.join(base_dir, 'std.csv'))
    return


def pre_process(base_dir, output_dir):
    warnings.filterwarnings('ignore', category=NaturalNameWarning)
    trade_dates = get_trade_dates(os.path.join(base_dir, 'data0'))
    folders = ['data0', 'data1', 'data2', 'data3', 'data4']
    t0 = time.time()
    for i, date in enumerate(trade_dates):
        yy, mm, dd = date.split('-')
        y = load_h5py(os.path.join(base_dir, 'y', yy, mm, dd, 'data.hdf'))
        y = y.set_index(['DateTime', 'Uid'])['y'].unstack()
        snap_times_map = get_snap_time(int(yy), int(mm), int(dd))
        data_paths = [os.path.join(base_dir, folder, yy, mm, dd, 'data.csv.gz') for folder in folders]
        dfs = [load_file(data_path) for data_path in data_paths if os.path.exists(data_path)]
        grouped_dfs = [df.groupby('SnapTime') for df in dfs]

        df_dict = {snap_time: _merge_dfs(grouped_dfs, snap_time) for snap_time in snap_times_map}
        out_path = os.path.join(output_dir, '-'.join([yy, mm, dd]) + '.hdf')

        h5store = HDFStore(out_path, 'w')
        for k, v in df_dict.items():
            h5store[k] = v
        h5store['y'] = y
        h5store.close()
        print("Progress: {0:3d} / {1:3d}. Time Spent {2:.2f} Min".format(i, len(trade_dates), (time.time() - t0) / 60))
    return


def _merge_dfs(dfs, k):
    valid_dfs = [df.get_group(k).drop(columns=['SnapTime']) for df in dfs if k in df.groups]
    if len(valid_dfs) == 0:
        return pd.DataFrame
    merged = reduce(lambda left, right: pd.merge(left, right, how='outer', on='Uid'), valid_dfs).set_index('Uid')
    valid_keys = [k for k, v in merged.dtypes.items() if 'X' in k and _valid_type(v)]
    return merged[valid_keys]


def check_date_sanity(base_dir):
    x_names = ['data0', 'data1', 'data2', 'data3', 'data4']
    x_dates = [get_trade_dates(os.path.join(base_dir, x_name)) for x_name in x_names]
    for i, x_date in enumerate(x_dates):
        assert len(set(x_date)) == len(x_date)
        assert len(set(x_date)) <= len(set(x_dates[0]))  # dataset 4 has fewer dates


def write_feat_uid(base_dir):
    res = []
    uid = set()
    for folder in ['data0', 'data1', 'data2', 'data3', 'data4']:
        feats = set()
        x_folder = os.path.join(base_dir, folder)
        files = iter_files(x_folder)
        t0 = time.time()
        for i, file in enumerate(files):
            data = pd.read_csv(file)
            uid = uid.union([uid for uid in list(data.Uid) if type(uid) is str])
            valid_feats = [k for k, v in data.dtypes.items() if 'X' in k and _valid_type(v)]
            feats = feats.union(valid_feats)
            if i % 1 == 0:
                print("Progress: {0:3d} / {1:3d}. "
                      "Time Spent {2:.2f} Min".format(i, len(files), (time.time() - t0) / 60))
        feats = sorted(list(feats))
        res.append(feats)
    uid = sorted(list(uid))
    res.append(uid)
    with open(os.path.join(base_dir, 'indexes.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(res)
    return
