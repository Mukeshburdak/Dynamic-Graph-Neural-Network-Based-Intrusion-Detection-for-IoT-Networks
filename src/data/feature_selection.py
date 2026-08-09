"""
feature_selection.py

Analyzes the 46 CICIoT2023 features for redundancy and selects a refined
feature subset to feed into graph construction / model training. Two
filters are applied, in order:

    1. Variance Threshold  -> drops near-constant features that carry
                               almost no discriminative information
                               (e.g. a protocol flag that is 0 in 99.9% of
                               rows across the whole balanced dataset).
    2. Correlation Filter   -> drops one feature from any pair with
                               |Pearson correlation| above a threshold
                               (default 0.95), since highly correlated
                               features add redundancy and can destabilize
                               GNN training without adding information.

The selected feature list is saved to
    data/processed/selected_features.json
so that graph_builder.py, models, and evaluation all use the exact same
reduced feature set. A correlation heatmap is also saved to
    results/figures/feature_correlation_heatmap.png
for the report/thesis write-up.
"""

import os
import json
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for headless runs
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold

from src import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

SELECTED_FEATURES_PATH = os.path.join(
    config.PROCESSED_DATA_DIR, "selected_features.json"
)


# ---------------------------------------------------------------------------
# 1. VARIANCE THRESHOLD FILTER
# ---------------------------------------------------------------------------

def apply_variance_threshold(
    df: pd.DataFrame, feature_columns: list, threshold: float = 0.01
) -> list:
    """
    Drop features whose variance is below `threshold`. Such features are
    (near) constant across the dataset and provide little to no signal for
    distinguishing attack categories.

    Parameters
    ----------
    df : pd.DataFrame
        Scaled or raw numeric feature DataFrame (must contain feature_columns).
    feature_columns : list
        Candidate feature column names to evaluate.
    threshold : float
        Minimum variance required to keep a feature.

    Returns
    -------
    list
        Feature names that passed the variance threshold, in original order.
    """
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(df[feature_columns])

    kept_mask = selector.get_support()
    kept_features = [f for f, keep in zip(feature_columns, kept_mask) if keep]
    dropped_features = [f for f, keep in zip(feature_columns, kept_mask) if not keep]

    if dropped_features:
        logger.info(
            f"Variance threshold ({threshold}) dropped {len(dropped_features)} "
            f"near-constant feature(s): {dropped_features}"
        )
    else:
        logger.info("Variance threshold dropped no features.")

    return kept_features


# ---------------------------------------------------------------------------
# 2. CORRELATION FILTER
# ---------------------------------------------------------------------------

def apply_correlation_filter(
    df: pd.DataFrame, feature_columns: list, corr_threshold: float = 0.95
) -> list:
    """
    Compute the pairwise Pearson correlation matrix over feature_columns and
    greedily drop one feature from any pair whose absolute correlation
    exceeds corr_threshold. For each correlated pair, the SECOND feature
    (in column order) is the one dropped, keeping the first occurrence.

    Parameters
    ----------
    df : pd.DataFrame
        Numeric feature DataFrame.
    feature_columns : list
        Candidate feature column names (typically the output of
        apply_variance_threshold()).
    corr_threshold : float
        Absolute correlation above which a feature is considered redundant.

    Returns
    -------
    list
        Feature names remaining after correlation filtering.
    """
    corr_matrix = df[feature_columns].corr().abs()

    # Only look at the upper triangle to avoid comparing a pair twice or
    # comparing a feature against itself.
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = [
        column
        for column in upper_triangle.columns
        if any(upper_triangle[column] > corr_threshold)
    ]

    kept_features = [f for f in feature_columns if f not in to_drop]

    if to_drop:
        logger.info(
            f"Correlation filter (>{corr_threshold}) dropped "
            f"{len(to_drop)} redundant feature(s): {to_drop}"
        )
    else:
        logger.info("Correlation filter dropped no features.")

    return kept_features, corr_matrix


# ---------------------------------------------------------------------------
# 3. VISUALIZATION
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(corr_matrix: pd.DataFrame, save_path: str = None):
    """
    Save a correlation heatmap of the given correlation matrix to disk for
    inclusion in the report/thesis.
    """
    if save_path is None:
        save_path = os.path.join(
            config.FIGURES_DIR, "feature_correlation_heatmap.png"
        )

    plt.figure(figsize=(16, 14))
    sns.heatmap(
        corr_matrix,
        cmap="coolwarm",
        vmin=0,
        vmax=1,
        square=True,
        xticklabels=True,
        yticklabels=True,
    )
    plt.title("Feature Correlation Heatmap (CICIoT2023)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    logger.info(f"Saved correlation heatmap to: {save_path}")


# ---------------------------------------------------------------------------
# 4. SAVE / LOAD SELECTED FEATURES
# ---------------------------------------------------------------------------

def save_selected_features(feature_list: list, path: str = SELECTED_FEATURES_PATH):
    """Persist the final selected feature list to a JSON file on disk."""
    with open(path, "w") as f:
        json.dump(feature_list, f, indent=2)
    logger.info(f"Saved {len(feature_list)} selected features to: {path}")


def get_selected_features(path: str = SELECTED_FEATURES_PATH) -> list:
    """
    Load the previously saved selected feature list. If the file does not
    exist yet (feature_selection.py has not been run), falls back to the
    full config.FEATURE_COLUMNS list so downstream modules never crash.
    """
    if os.path.exists(path):
        with open(path, "r") as f:
            feature_list = json.load(f)
        logger.info(f"Loaded {len(feature_list)} selected features from: {path}")
        return feature_list

    logger.warning(
        f"No selected_features.json found at {path}. Falling back to the "
        f"full {len(config.FEATURE_COLUMNS)}-feature set from config.py. "
        f"Run feature_selection.py to generate a reduced feature set."
    )
    return list(config.FEATURE_COLUMNS)


# ---------------------------------------------------------------------------
# 5. FULL PIPELINE ENTRY POINT
# ---------------------------------------------------------------------------

def run_feature_selection(
    variance_threshold: float = 0.01,
    corr_threshold: float = 0.95,
):
    """
    Run the full feature selection pipeline using the already-preprocessed
    training split (config.TRAIN_CSV_PATH), save the resulting reduced
    feature list to disk, and save a correlation heatmap figure.

    Requires that preprocessing.run_preprocessing() has already been run,
    since this reads train.csv rather than the raw dataset.
    """
    if not os.path.exists(config.TRAIN_CSV_PATH):
        raise FileNotFoundError(
            f"{config.TRAIN_CSV_PATH} not found. Run "
            f"src/data/preprocessing.py first to generate train.csv."
        )

    logger.info(f"Loading training data from: {config.TRAIN_CSV_PATH}")
    train_df = pd.read_csv(config.TRAIN_CSV_PATH)

    logger.info("STEP 1/3: Applying variance threshold filter...")
    features_after_variance = apply_variance_threshold(
        train_df, config.FEATURE_COLUMNS, threshold=variance_threshold
    )

    logger.info("STEP 2/3: Applying correlation filter...")
    final_features, corr_matrix = apply_correlation_filter(
        train_df, features_after_variance, corr_threshold=corr_threshold
    )

    logger.info("STEP 3/3: Saving results...")
    plot_correlation_heatmap(corr_matrix)
    save_selected_features(final_features)

    logger.info(
        f"Feature selection complete: {len(config.FEATURE_COLUMNS)} -> "
        f"{len(final_features)} features."
    )
    return final_features


# ---------------------------------------------------------------------------
# Manual run: `python -m src.data.feature_selection` from the project root.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_feature_selection()
