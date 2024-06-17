from pathlib import Path
import pickle

import pandas as pd
from sklearn import linear_model

from util import SampleType, PrepType, DataType, get_file_name


if __name__ == "__main__":
    name = "reg1"
    X_train = pd.read_csv(
        get_file_name(data_name=name, sample_type=SampleType.TRAIN, prep_type=PrepType.PREPARED, data_type=DataType.X)
    )
    y_train = pd.read_csv(
        get_file_name(data_name=name, sample_type=SampleType.TRAIN, prep_type=PrepType.PREPARED, data_type=DataType.Y)
    )

    estimator = linear_model.LinearRegression()
    estimator.fit(X_train, y_train)

    with open('estimator.pkl', 'wb') as f:
        pickle.dump(estimator, f)
