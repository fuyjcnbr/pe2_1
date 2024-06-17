from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    transformers_pre_li = [
        {
            "pipe": Pipeline([
                ('step_with_names', StandardScaler()),
            ]),
            "columns": df.columns,
        },
    ]

    transformers = []
    for v in transformers_pre_li:
      pipe = v["pipe"]
      for col in v["columns"]:
        tu = (f"cat_{col.lower()}", pipe, [col])
        transformers.append(tu)

    preprocessors = ColumnTransformer(transformers=transformers)
    preprocessors.fit(df)

    li = []
    for t in preprocessors.transformers_:
        if isinstance(t[1], Pipeline):
            old_name = t[1].feature_names_in_
            li.append(t[1]["step_with_names"].get_feature_names_out(old_name))
    new_columns = np.hstack(li)

    _df = preprocessors.transform(df)
    df_transformed_train = pd.DataFrame(_df, columns=[new_columns])
    return df_transformed_train


def generate_prepared_data(name: str):
    X_train = pd.read_csv(Path("train").joinpath(f"{name}_x_raw.csv"))
    y_train = pd.read_csv(Path("train").joinpath(f"{name}_y_raw.csv"))

    X_test = pd.read_csv(Path("test").joinpath(f"{name}_x_raw.csv"))
    y_test = pd.read_csv(Path("test").joinpath(f"{name}_y_raw.csv"))

    X_train_prep = prepare_dataframe(X_train)
    y_train_prep = prepare_dataframe(y_train)
    X_test_prep = prepare_dataframe(X_test)
    y_test_prep = prepare_dataframe(y_test)

    X_train_prep.to_csv(Path("train").joinpath(f"{name}_x_prep.csv"))
    y_train_prep.to_csv(Path("train").joinpath(f"{name}_y_prep.csv"))

    X_test_prep.to_csv(Path("test").joinpath(f"{name}_x_prep.csv"))
    y_test_prep.to_csv(Path("test").joinpath(f"{name}_y_prep.csv"))


if __name__ == "__main__":
    generate_prepared_data("reg1")
