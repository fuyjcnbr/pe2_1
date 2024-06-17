from enum import Enum
from pathlib import Path


class SampleType(Enum):
    TRAIN = 1
    TEST = 2


class PrepType(Enum):
    RAW = 1
    PREPARED = 2


class DataType(Enum):
    X = 1
    Y = 2


def get_file_name(data_name: str, sample_type: SampleType, prep_type: PrepType, data_type:DataType) -> Path:
    if sample_type == SampleType.TEST:
        dir_str = "test"
    elif sample_type == SampleType.TRAIN:
        dir_str = "train"
    else:
        dir_str = "unknown"

    if prep_type == PrepType.RAW:
        prep_str = "raw"
    elif prep_type == PrepType.PREPARED:
        prep_str = "prep"
    else:
        prep_str = "unknown"

    if data_type == DataType.X:
        data_type_str = "x"
    elif data_type == DataType.Y:
        data_type_str = "y"
    else:
        data_type_str = "unknown"

    p = Path(dir_str).joinpath(f"{data_name}_{data_type_str}_{prep_str}.csv")
    return p
