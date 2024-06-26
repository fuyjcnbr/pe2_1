from pathlib import Path
from random import randint

import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


def generate_dataset(name: str):
    X, y = make_regression(n_samples=1000, n_features=10, noise=1, random_state=randint(100, 900))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=randint(100, 900))
    columns = [f"col_{i}" for i in list(range(X.shape[1]))]

    X_train_df = pd.DataFrame(data=X_train, columns=columns)
    X_train_df.to_csv(Path("train").joinpath(f"{name}_x_raw.csv"), index=False)

    y_train_df = pd.DataFrame(data=y_train, columns=["y"])
    y_train_df.to_csv(Path("train").joinpath(f"{name}_y_raw.csv"), index=False)

    X_test_df = pd.DataFrame(data=X_test, columns=columns)
    X_test_df.to_csv(Path("test").joinpath(f"{name}_x_raw.csv"), index=False)

    y_test_df = pd.DataFrame(data=y_test, columns=["y"])
    y_test_df.to_csv(Path("test").joinpath(f"{name}_y_raw.csv"), index=False)


if __name__ == "__main__":
    Path("train").mkdir(parents=True, exist_ok=True)
    Path("test").mkdir(parents=True, exist_ok=True)
    generate_dataset("reg1")
