import csv
from pathlib import Path

import numpy as np
from svalbard.data_model.data_file import Data
from svalbard.utility.datasets_from_csv import datasets_from_csv


def test_datasets_from_csv():
    fields = ["test1", "test2", "test3"]
    rows = [[1, 2, 3], [11, 22, 33], [111, 222, 333], [1111, 2222, 3333]]
    path = Path("test_results.csv")
    with open(path, "w") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        csvwriter.writerows(rows)
    datasets = datasets_from_csv(path)
    assert len(datasets) == 3
    for col, dataset in zip(fields, datasets):
        assert isinstance(dataset, Data.DataSet)
        assert dataset.name == col
        assert dataset.memory.shape == (4,)
        assert dataset.memory.dtype == "int64"
        assert dataset.memory.to_array().shape == (4,)
        assert dataset.memory.to_array().dtype == np.int64
    path.unlink()
