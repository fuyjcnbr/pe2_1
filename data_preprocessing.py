from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from util import SampleType, PrepType, DataType, get_file_name


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
    X_train = pd.read_csv(
        get_file_name(data_name=name, sample_type=SampleType.TRAIN, prep_type=PrepType.RAW, data_type=DataType.X)
    )
    y_train = pd.read_csv(
        get_file_name(data_name=name, sample_type=SampleType.TRAIN, prep_type=PrepType.RAW, data_type=DataType.Y)
    )

    X_test = pd.read_csv(
        get_file_name(data_name=name, sample_type=SampleType.TEST, prep_type=PrepType.RAW, data_type=DataType.X)
    )
    y_test = pd.read_csv(
        get_file_name(data_name=name, sample_type=SampleType.TEST, prep_type=PrepType.RAW, data_type=DataType.Y)
    )

    X_train_prep = prepare_dataframe(X_train)
    y_train_prep = prepare_dataframe(y_train)
    X_test_prep = prepare_dataframe(X_test)
    y_test_prep = prepare_dataframe(y_test)

    X_train_prep.to_csv(
        get_file_name(data_name=name, sample_type=SampleType.TRAIN, prep_type=PrepType.PREPARED, data_type=DataType.X),
        index=False,
    )
    y_train_prep.to_csv(
        get_file_name(data_name=name, sample_type=SampleType.TRAIN, prep_type=PrepType.PREPARED, data_type=DataType.Y),
        index=False,
    )

    X_test_prep.to_csv(
        get_file_name(data_name=name, sample_type=SampleType.TEST, prep_type=PrepType.PREPARED, data_type=DataType.X),
        index=False,
    )
    y_test_prep.to_csv(
        get_file_name(data_name=name, sample_type=SampleType.TEST, prep_type=PrepType.PREPARED, data_type=DataType.Y),
        index=False,
    )


if __name__ == "__main__":
    generate_prepared_data("reg1")
