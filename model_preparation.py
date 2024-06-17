from pathlib import Path
import pickle

import pandas as pd
from sklearn import linear_model


if __name__ == "__main__":
    name = "reg1"
    X_train = pd.read_csv(Path("train").joinpath(f"{name}_x_prep.csv"))
    y_train = pd.read_csv(Path("train").joinpath(f"{name}_y_prep.csv"))

    estimator = linear_model.LinearRegression()
    estimator.fit(X_train, y_train)

    with open('estimator.pkl', 'wb') as f:
        pickle.dump(estimator, f)
