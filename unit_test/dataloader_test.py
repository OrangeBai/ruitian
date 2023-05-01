import unittest
from dataloader.loader import load_file, load_h5py


class TestDataLoader(unittest.TestCase):
    def test_load_file(self) -> None:
        data = load_file(r"E:\Dataset\Ruitian\09628_basefeature_candidate\data1\2018\01\02\data.csv.gz")
        return data

    def test_load_h5py(self) -> None:
        data = load_h5py(r"E:\Dataset\Ruitian\09628_basefeature_candidate\y\2018\07\03\data.hdf")
        # return data
        return


if __name__ == '__main__':
    unittest.main()
