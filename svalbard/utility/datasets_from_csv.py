"""Method for creating a list of DataSets from a csv file."""
from pathlib import Path

import pandas as pd
from svalbard.data_model.data_file import Data


def datasets_from_csv(results: Path) -> list[Data.DataSet]:
    """
    Creates a list of DataSets from a csv file.

    Args:
        results (Path):
            The path to the local csv file of results. Columns should be all
            values for a given parameter in the sweep.

    Returns:
        list[Data.DataSet]:
            A list of datasets, where each dataset contains all values for a
            given parameter in the sweep.
    """
    df = pd.read_csv(results)
    return [
        Data.DataSet.from_array(name=col, array=values) for (col, values) in df.items()
    ]
