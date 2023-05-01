import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils.external import get_trade_dates, load_feat_uid, get_snap_time
import os
import torch
import time
from pandas import HDFStore


class OpeningData(Dataset):
    def __init__(self,
                 base_dir,
                 proc_dir,
                 input_uid,
                 target_uid,
                 folders,
                 period):
        self.base_dir = base_dir
        self.proc_dir = proc_dir
        self.folders = folders
        self.feats, self.uid = load_feat_uid(self.base_dir, self.folders)

        self.input_uid = input_uid if input_uid is not None else self.uid
        self.target_uid = target_uid if target_uid is not None else self.uid

        self.period = period
        self.trade_dates = get_trade_dates(os.path.join(base_dir, 'data0'), *period)
        self.mean_std = pd.read_csv(os.path.join(self.base_dir, 'mean_std.csv'), index_col=0)

    def __len__(self):
        return len(self.trade_dates)

    def __getitem__(self, index):
        cur_date = self.trade_dates[index]
        yy, mm, dd = cur_date.split('-')

        # Load X from data store
        os.path.join(self.proc_dir)
        snap_times = get_snap_time(int(yy), int(mm), int(dd))
        store = HDFStore(os.path.join(self.proc_dir, cur_date + '.hdf'))
        load_x = [self._select(store[snap_time]) for snap_time in snap_times]
        y = store['y']
        store.close()

        # pre process
        y = pd.DataFrame(y, index=snap_times, columns=self.target_uid)
        y = y.loc[:, self.target_uid].to_numpy(dtype=np.float32)
        x, y = np.array(load_x), np.array(y)
        return x, y

    def _select(self, x):
        return pd.DataFrame(x, index=self.input_uid, columns=self.feats)


class Normalize(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, input_x):
        normed_x = (input_x - self.mean) / self.std
        return normed_x


def set_dataloader(base_dir, proc_dir, input_uid, target_uid, folders, period, batch_size=1, num_workers=8):
    dataset = OpeningData(
        base_dir=base_dir,
        proc_dir=proc_dir,
        input_uid=input_uid,
        target_uid=target_uid,
        folders=folders,
        period=period,
    )

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, prefetch_factor=8)


if __name__ == '__main__':
    pass
    # open_dataset = OpeningData(
    #     base_dir=r"E:\Dataset\Ruitian\09628_basefeature_candidate",
    #     proc_dir=r"D:\ruitian\processed",
    #     target_uid=[r"000001-SZ-stock"],
    #     folders=['data0', 'data1', 'data2', 'data3', 'data4'],
    #     period=('2018-01-01', '2019-07-01')
    # )
    # t0 = time.time()
    # for i in range(len(open_dataset)):
    #     x, _ = open_dataset.__getitem__(i)
    #     print(f'{i}: max: {x.max()}, min: {x.min()}, '
    #           f'nan: {np.sum(np.isnan(x), axis=(0, 1, 2))}, '
    #           f'time: {time.time() - t0}')
