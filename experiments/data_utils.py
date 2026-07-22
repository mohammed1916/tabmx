"""Shared dataset loading with proper categorical handling.

Loads OpenML datasets as DataFrames, ordinal-encodes categorical/string columns
(TabPFN handles categoricals natively given their indices), leaves numeric
columns as float, and returns (X, y, categorical_indices).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


def load_openml(openml_id: int):
    """Return (X_float, y_int, categorical_indices).

    - Categorical / string / boolean columns are ordinal-encoded to floats.
      Their column indices are returned so TabPFN can treat them as categorical.
    - Numeric columns are coerced to float; non-numeric tokens -> NaN
      (TabPFN handles missing values).
    - NaN-category cells are left as NaN after encoding.
    """
    X_df, y = fetch_openml(
        data_id=openml_id, as_frame=True, return_X_y=True, parser="auto"
    )

    cat_cols = []
    num_cols = []
    for col in X_df.columns:
        dtype = X_df[col].dtype
        if (
            isinstance(dtype, pd.CategoricalDtype)
            or dtype == object
            or dtype == bool
            or str(dtype) == "category"
        ):
            cat_cols.append(col)
        else:
            num_cols.append(col)

    n_cols = len(X_df.columns)
    X_out = np.empty((len(X_df), n_cols), dtype=np.float64)
    cat_indices = []

    for j, col in enumerate(X_df.columns):
        if col in cat_cols:
            enc = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=np.nan,
                encoded_missing_value=np.nan,
            )
            vals = X_df[[col]].astype("object")
            X_out[:, j] = enc.fit_transform(vals).ravel()
            cat_indices.append(j)
        else:
            X_out[:, j] = pd.to_numeric(X_df[col], errors="coerce").to_numpy(
                dtype=np.float64
            )

    y_enc = LabelEncoder().fit_transform(np.asarray(y).astype(str))
    return X_out, y_enc, cat_indices
