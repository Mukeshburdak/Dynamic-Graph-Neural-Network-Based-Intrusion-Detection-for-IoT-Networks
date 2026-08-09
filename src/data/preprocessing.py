"""
preprocessing.py

Takes the raw, merged DataFrame produced by loader.py and turns it into
clean, numeric, balanced, scaled train/validation/test CSVs ready for graph
construction (graph_builder.py) and model training.

Pipeline steps (in order):
    1. Clean       -> drop duplicates, handle inf/NaN values
    2. Encode       -> map string 'Category' labels to integer 'Label'
                        using config.LABEL_TO_INDEX
    3. Balance      -> reduce severe class imbalance via a combination of
                        undersampling the majority class(es) and
                        oversampling rare class(es)
    4. Split        -> stratified train/validation/test split
    5. Scale        -> StandardScaler fit ONLY on the train split, then
                        applied to validation/test (prevents data leakage)
    6. Save         -> write train.csv / validation.csv / test.csv to
                        config.PROCESSED_DATA_DIR, and persist the fitted
                        scaler to disk for later reuse at inference time
"""

import os
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import joblib

from src import config
from src.data.loader import load_all_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. CLEANING
# ---------------------------------------------------------------------------

def clean_data(df: pd.DataFrame, feature_columns: list = None) -> pd.DataFrame:
    """
    Clean the raw merged DataFrame:
      - Coerce all feature columns to numeric (invalid parses -> NaN).
      - Replace +/-inf (common in rate-based features when duration = 0)
        with NaN.
      - Drop rows where ANY feature is NaN (safer than imputing for a
        security dataset, since fabricated values could distort attack
        signatures).
      - Drop exact duplicate rows.

    Parameters
    ----------
    df : pd.DataFrame
        Raw combined DataFrame from loader.load_all_data(), containing
        config.FEATURE_COLUMNS + 'Category'.
    feature_columns : list, optional
        Which columns to treat as numeric features. Defaults to the full
        config.FEATURE_COLUMNS (production use). Exposed as a parameter so
        this function can also be tested/used against a smaller or
        alternative column set without editing config.py.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with the same columns.
    """
    if feature_columns is None:
        feature_columns = config.FEATURE_COLUMNS

    initial_rows = len(df)

    # Ensure every expected feature column is present; anything missing is
    # a sign the raw schema drifted from the expected column set.
    missing_cols = [c for c in feature_columns if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"The following expected feature columns are missing from the "
            f"loaded data: {missing_cols}. Check config.FEATURE_COLUMNS "
            f"against the actual CSV header."
        )

    # Coerce features to numeric.
    for col in feature_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Replace infinite values (can occur in Rate/Srate/Drate when a flow's
    # duration is extremely small or zero) with NaN so they get dropped
    # along with other missing values below.
    df[feature_columns] = df[feature_columns].replace([np.inf, -np.inf], np.nan)

    # Drop rows with any missing feature value.
    df = df.dropna(subset=feature_columns)

    # Drop exact duplicate rows (identical features + label).
    df = df.drop_duplicates()

    df = df.reset_index(drop=True)

    dropped = initial_rows - len(df)
    logger.info(
        f"Cleaning complete: {initial_rows:,} -> {len(df):,} rows "
        f"({dropped:,} dropped, {dropped / initial_rows:.2%})"
    )
    return df


# ---------------------------------------------------------------------------
# 2. LABEL ENCODING
# ---------------------------------------------------------------------------

def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map the string 'Category' column (e.g. 'DDoS', 'Benign') to an integer
    'Label' column using config.LABEL_TO_INDEX, so the value is ready for
    PyTorch's CrossEntropyLoss.
    """
    unknown_categories = set(df["Category"].unique()) - set(config.CLASS_NAMES)
    if unknown_categories:
        raise ValueError(
            f"Found categories not present in config.CLASS_NAMES: "
            f"{unknown_categories}. Update config.py before proceeding."
        )

    df["Label"] = df["Category"].map(config.LABEL_TO_INDEX)
    return df


# ---------------------------------------------------------------------------
# 3. CLASS BALANCING
# ---------------------------------------------------------------------------

def balance_classes(
    df: pd.DataFrame,
    max_per_class: int = 50000,
    min_per_class: int = 5000,
    random_state: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """
    Reduce the severe class imbalance documented for CICIoT2023 (ratios up
    to ~5,751:1 between the most and least frequent classes) using a
    combined strategy:

      - UNDERSAMPLE any class with more than `max_per_class` rows down to
        exactly `max_per_class` (random sampling without replacement).
      - OVERSAMPLE any class with fewer than `min_per_class` rows up to
        exactly `min_per_class` (random sampling WITH replacement).
      - Classes already between min_per_class and max_per_class are left
        untouched.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned, label-encoded DataFrame (must contain 'Category').
    max_per_class : int
        Upper cap applied per category via undersampling.
    min_per_class : int
        Lower floor applied per category via oversampling.
    random_state : int
        Seed for reproducible resampling.

    Returns
    -------
    pd.DataFrame
        Balanced DataFrame, shuffled.
    """
    balanced_frames = []

    for category, group in df.groupby("Category"):
        n = len(group)

        if n > max_per_class:
            group = resample(
                group,
                replace=False,
                n_samples=max_per_class,
                random_state=random_state,
            )
            logger.info(f"  {category}: {n:,} -> {max_per_class:,} (undersampled)")
        elif n < min_per_class:
            group = resample(
                group,
                replace=True,
                n_samples=min_per_class,
                random_state=random_state,
            )
            logger.info(f"  {category}: {n:,} -> {min_per_class:,} (oversampled)")
        else:
            logger.info(f"  {category}: {n:,} (unchanged)")

        balanced_frames.append(group)

    balanced_df = pd.concat(balanced_frames, ignore_index=True)

    # Shuffle so classes aren't grouped together in sequence (important
    # since graph_builder.py will later slide a window over row order).
    balanced_df = balanced_df.sample(
        frac=1.0, random_state=random_state
    ).reset_index(drop=True)

    logger.info(f"Balancing complete. Total rows after balancing: {len(balanced_df):,}")
    return balanced_df


# ---------------------------------------------------------------------------
# 4. TRAIN / VALIDATION / TEST SPLIT
# ---------------------------------------------------------------------------

def split_data(df: pd.DataFrame):
    """
    Stratified split into train/validation/test sets using the ratios
    defined in config.py (default 70/15/15), preserving class proportions
    in every split.

    Returns
    -------
    (train_df, val_df, test_df) : tuple of pd.DataFrame
    """
    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - config.TRAIN_RATIO),
        stratify=df["Label"],
        random_state=config.RANDOM_SEED,
    )

    # Split the remaining (val + test) portion in half, proportionally.
    relative_val_ratio = config.VAL_RATIO / (config.VAL_RATIO + config.TEST_RATIO)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - relative_val_ratio),
        stratify=temp_df["Label"],
        random_state=config.RANDOM_SEED,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    logger.info(
        f"Split sizes -> train: {len(train_df):,} | "
        f"validation: {len(val_df):,} | test: {len(test_df):,}"
    )
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# 5. FEATURE SCALING
# ---------------------------------------------------------------------------

def scale_features(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Fit a StandardScaler on the TRAIN split only (to avoid data leakage from
    validation/test statistics), then transform all three splits.

    The fitted scaler is saved to disk so the exact same transformation can
    be reapplied later (e.g. during real-time inference on new traffic).

    Returns
    -------
    (train_df, val_df, test_df) : tuple of pd.DataFrame
        Same DataFrames with feature columns replaced by their scaled
        values. Non-feature columns ('Category', 'Label') are left as-is.
    """
    scaler = StandardScaler()

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df[config.FEATURE_COLUMNS] = scaler.fit_transform(
        train_df[config.FEATURE_COLUMNS]
    )
    val_df[config.FEATURE_COLUMNS] = scaler.transform(val_df[config.FEATURE_COLUMNS])
    test_df[config.FEATURE_COLUMNS] = scaler.transform(test_df[config.FEATURE_COLUMNS])

    scaler_path = os.path.join(config.PROCESSED_DATA_DIR, "feature_scaler.joblib")
    joblib.dump(scaler, scaler_path)
    logger.info(f"Saved fitted StandardScaler to: {scaler_path}")

    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# 6. FULL PIPELINE ENTRY POINT
# ---------------------------------------------------------------------------

def run_preprocessing(
    nrows_per_file: int = None,
    max_per_class: int = 50000,
    min_per_class: int = 5000,
):
    """
    Run the full preprocessing pipeline end-to-end and write the final
    train.csv / validation.csv / test.csv files to config.PROCESSED_DATA_DIR.

    Parameters
    ----------
    nrows_per_file : int, optional
        Passed through to loader.load_all_data() to allow quick testing on
        a small sample instead of the full ~13GB raw dataset.
    max_per_class : int
        Upper cap per category for balance_classes().
    min_per_class : int
        Lower floor per category for balance_classes().
    """
    logger.info("STEP 1/6: Loading raw data...")
    df = load_all_data(nrows_per_file=nrows_per_file, save_merged_csv=False)

    logger.info("STEP 2/6: Cleaning data...")
    df = clean_data(df)

    logger.info("STEP 3/6: Encoding labels...")
    df = encode_labels(df)

    logger.info("STEP 4/6: Balancing classes...")
    df = balance_classes(
        df, max_per_class=max_per_class, min_per_class=min_per_class
    )

    logger.info("STEP 5/6: Splitting into train/validation/test...")
    train_df, val_df, test_df = split_data(df)

    logger.info("STEP 6/6: Scaling features...")
    train_df, val_df, test_df = scale_features(train_df, val_df, test_df)

    train_df.to_csv(config.TRAIN_CSV_PATH, index=False)
    val_df.to_csv(config.VAL_CSV_PATH, index=False)
    test_df.to_csv(config.TEST_CSV_PATH, index=False)

    logger.info(f"Saved train.csv -> {config.TRAIN_CSV_PATH} ({len(train_df):,} rows)")
    logger.info(f"Saved validation.csv -> {config.VAL_CSV_PATH} ({len(val_df):,} rows)")
    logger.info(f"Saved test.csv -> {config.TEST_CSV_PATH} ({len(test_df):,} rows)")
    logger.info("Preprocessing pipeline complete.")

    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Manual run: `python -m src.data.preprocessing` from the project root.
# Uses a small per-file row sample by default for a fast smoke test.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_preprocessing(nrows_per_file=2000, max_per_class=2000, min_per_class=200)
