from pathlib import Path
import pickle

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from util import SampleType, PrepType, DataType, get_file_name

if __name__ == "__main__":
    name = "reg1"
    X_test = pd.read_csv(
        get_file_name(data_name=name, sample_type=SampleType.TEST, prep_type=PrepType.PREPARED, data_type=DataType.X)
    )
    y_test = pd.read_csv(
        get_file_name(data_name=name, sample_type=SampleType.TEST, prep_type=PrepType.PREPARED, data_type=DataType.Y)
    )

    with open('estimator.pkl', 'rb') as f:
        estimator = pickle.load(f)

    y_predicted = estimator.predict(X_test)

    print(f"MSE на тестовой выборке: {mean_squared_error(y_predicted, y_test):.4f}")
    print(f"r2 на тестовой выборке: {r2_score(y_predicted, y_test):.4f}")
