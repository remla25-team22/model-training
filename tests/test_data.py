import pytest
import pandas as pd
import numpy as np
import joblib

# Define expected schema and thresholds
EXPECTED_COLUMNS = {
    'cleaned': {'dtype': object, 'non_null': True},
    'Liked': {'dtype': int, 'allowed_values': [0, 1]}
}
LENGTH_THRESHOLD = {'min': 0, 'max': 500}

@pytest.mark.ml_test("FD-1")   # Data-1: feature expectations captured in a schema
@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_data_invariants(split):
    path = f"data/preprocessed/{split}.csv"
    df = pd.read_csv(path, keep_default_na=False)

    for col, rules in EXPECTED_COLUMNS.items():
        assert col in df.columns, f"Column '{col}' missing in {split}.csv"

        if rules.get('non_null'):
            null_rows = df[df[col].isnull()]
            if not null_rows.empty:
                print(f"\n Null values found in column '{col}' of {split}.csv:")
                print(null_rows)
            assert null_rows.empty, f"Null values found in '{col}' of {split}.csv"

        if rules['dtype'] is int:
            assert pd.api.types.is_integer_dtype(df[col]), f"Column '{col}' is not integer type in {split}.csv"
        elif rules['dtype'] is object:
            assert pd.api.types.is_string_dtype(df[col]), f"Column '{col}' is not string type in {split}.csv"

        if 'allowed_values' in rules:
            invalid = set(df[col].unique()) - set(rules['allowed_values'])
            assert not invalid, f"Invalid values {invalid} found in '{col}' of {split}.csv"

    lengths = df['cleaned'].str.len()
    assert lengths.min() >= LENGTH_THRESHOLD['min'], f"Some 'cleaned' entries are smaller than {LENGTH_THRESHOLD['min']}"
    assert lengths.max() <= LENGTH_THRESHOLD['max'], f"Some 'cleaned' entries exceed {LENGTH_THRESHOLD['max']}"


@pytest.mark.ml_test("FD-2")   # Data-1:  All features are beneficial
def test_feature_benefit_by_coefficients():
    model = joblib.load('models/c2_model.pkl')
    coefs = model.coef_.ravel()         
    abs_coefs = np.abs(coefs)
    threshold = 1e-3
    n_useful = np.sum(abs_coefs > threshold)
    assert n_useful >= 1000 * 0.8, f"Too few useful features by coefficients: {n_useful}"
