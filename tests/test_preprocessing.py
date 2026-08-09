"""
tests/test_preprocessing.py

Unit tests for src/data/preprocessing.py. Uses small, hand-built synthetic
DataFrames instead of the real CICIoT2023 dataset, so these tests run in
milliseconds and don't require the raw data to be downloaded.
"""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data.preprocessing import (
    clean_data,
    encode_labels,
    balance_classes,
    split_data,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_feature_columns():
    """Use a small subset of real feature names for fast, readable tests."""
    return config.FEATURE_COLUMNS[:5]


@pytest.fixture
def clean_sample_df(sample_feature_columns):
    """A small, already-valid DataFrame with no NaN/inf and known categories."""
    n_rows = 40
    rng = np.random.default_rng(config.RANDOM_SEED)

    data = {col: rng.normal(size=n_rows) for col in sample_feature_columns}
    # 4 categories, 10 rows each, so we have a predictable distribution.
    data["Category"] = (
        ["Benign"] * 10 + ["DDoS"] * 10 + ["DoS"] * 10 + ["Recon"] * 10
    )
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# clean_data()
# ---------------------------------------------------------------------------

def test_clean_data_drops_nan_and_inf(sample_feature_columns):
    df = pd.DataFrame(
        {
            **{col: [1.0, 2.0, np.nan, 4.0, np.inf] for col in sample_feature_columns},
            "Category": ["Benign"] * 5,
        }
    )

    cleaned = clean_data(df, feature_columns=sample_feature_columns)

    # Rows with NaN or inf in ANY feature column should be dropped.
    # Only rows at index 0, 1, 3 have fully valid values.
    assert len(cleaned) == 3
    assert not cleaned[sample_feature_columns].isin([np.inf, -np.inf]).any().any()
    assert not cleaned[sample_feature_columns].isna().any().any()


def test_clean_data_drops_duplicates(sample_feature_columns):
    row = {col: 1.0 for col in sample_feature_columns}
    row["Category"] = "Benign"
    df = pd.DataFrame([row, row, row])  # 3 identical rows

    cleaned = clean_data(df, feature_columns=sample_feature_columns)
    assert len(cleaned) == 1


def test_clean_data_raises_on_missing_columns():
    df = pd.DataFrame({"some_other_column": [1, 2, 3], "Category": ["Benign"] * 3})
    with pytest.raises(ValueError):
        clean_data(df)


# ---------------------------------------------------------------------------
# encode_labels()
# ---------------------------------------------------------------------------

def test_encode_labels_maps_correctly(clean_sample_df):
    encoded = encode_labels(clean_sample_df)

    assert "Label" in encoded.columns
    for _, row in encoded.iterrows():
        assert row["Label"] == config.LABEL_TO_INDEX[row["Category"]]


def test_encode_labels_raises_on_unknown_category(sample_feature_columns):
    df = pd.DataFrame(
        {
            **{col: [1.0, 2.0] for col in sample_feature_columns},
            "Category": ["NotARealCategory", "Benign"],
        }
    )
    with pytest.raises(ValueError):
        encode_labels(df)


# ---------------------------------------------------------------------------
# balance_classes()
# ---------------------------------------------------------------------------

def test_balance_classes_undersamples_majority(sample_feature_columns):
    # 100 Benign rows (majority), 5 DDoS rows (minority).
    n_benign, n_ddos = 100, 5
    data = {
        col: np.concatenate(
            [np.random.randn(n_benign), np.random.randn(n_ddos)]
        )
        for col in sample_feature_columns
    }
    data["Category"] = ["Benign"] * n_benign + ["DDoS"] * n_ddos
    df = pd.DataFrame(data)

    balanced = balance_classes(df, max_per_class=20, min_per_class=10)

    counts = balanced["Category"].value_counts()
    assert counts["Benign"] == 20   # undersampled down to max_per_class
    assert counts["DDoS"] == 10     # oversampled up to min_per_class


def test_balance_classes_leaves_mid_range_untouched(sample_feature_columns):
    n = 15  # between min_per_class=10 and max_per_class=20
    data = {col: np.random.randn(n) for col in sample_feature_columns}
    data["Category"] = ["Benign"] * n
    df = pd.DataFrame(data)

    balanced = balance_classes(df, max_per_class=20, min_per_class=10)
    assert len(balanced) == n


# ---------------------------------------------------------------------------
# split_data()
# ---------------------------------------------------------------------------

def test_split_data_ratios_and_no_overlap(sample_feature_columns):
    n_per_class = 100
    data = {
        col: np.random.randn(n_per_class * 2) for col in sample_feature_columns
    }
    data["Category"] = ["Benign"] * n_per_class + ["DDoS"] * n_per_class
    df = pd.DataFrame(data)
    df = encode_labels(df)

    train_df, val_df, test_df = split_data(df)

    total = len(train_df) + len(val_df) + len(test_df)
    assert total == len(df)

    # Ratios should roughly match config (within a small tolerance since
    # stratified splitting on a small dataset can't be exact).
    assert abs(len(train_df) / total - config.TRAIN_RATIO) < 0.05
    assert abs(len(val_df) / total - config.VAL_RATIO) < 0.05
    assert abs(len(test_df) / total - config.TEST_RATIO) < 0.05

    # No overlapping indices between splits (using a unique identifier
    # column to check disjointness instead of relying on reset indices).
    train_ids = set(map(tuple, train_df[sample_feature_columns].values))
    test_ids = set(map(tuple, test_df[sample_feature_columns].values))
    assert train_ids.isdisjoint(test_ids)
