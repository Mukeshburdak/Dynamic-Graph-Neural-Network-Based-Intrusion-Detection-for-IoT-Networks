"""
loader.py

Loads the raw CICIoT2023 dataset from disk. The raw dataset is organized as
one sub-folder per individual attack (e.g. "DDoS-UDP_Flood/", "XSS/",
"Benign_Final/"), and each sub-folder contains one or more CSV part-files
with the 46 flow-level features described in config.FEATURE_COLUMNS.

This module is responsible ONLY for reading the raw CSVs and attaching the
correct 7-category (+Benign) label to every row. Cleaning, scaling, and
train/val/test splitting happen later in preprocessing.py.
"""

import os
import glob
import logging

import pandas as pd

from src import config

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def list_attack_folders(raw_dir: str = config.RAW_DATA_DIR) -> list:
    """
    Return a sorted list of full paths to every attack sub-folder inside
    raw_dir (e.g. data/raw/CICIoT2023/DDoS-UDP_Flood, .../XSS, etc.).

    Only folders that are also present as keys in
    config.ATTACK_FOLDER_TO_CATEGORY are returned, so any unexpected or
    extra folder in the raw data directory is safely ignored (with a
    warning) instead of silently breaking the pipeline.
    """
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(
            f"Raw data directory not found: {raw_dir}\n"
            f"Make sure you have downloaded and extracted the CICIoT2023 "
            f"CSV dataset into this location."
        )

    all_entries = sorted(os.listdir(raw_dir))
    valid_folders = []

    for entry in all_entries:
        full_path = os.path.join(raw_dir, entry)
        if not os.path.isdir(full_path):
            continue  # skip stray files like README_CSV.pdf

        if entry not in config.ATTACK_FOLDER_TO_CATEGORY:
            logger.warning(
                f"Folder '{entry}' has no entry in "
                f"config.ATTACK_FOLDER_TO_CATEGORY and will be SKIPPED. "
                f"Add it to the mapping in src/config.py if it should be "
                f"included."
            )
            continue

        valid_folders.append(full_path)

    if not valid_folders:
        raise RuntimeError(
            f"No valid attack folders found inside {raw_dir}. "
            f"Check that config.ATTACK_FOLDER_TO_CATEGORY matches your "
            f"actual folder names."
        )

    return valid_folders


def _find_csv_files(folder_path: str) -> list:
    """
    Return a sorted list of all .csv file paths directly inside folder_path.
    CICIoT2023 attack folders typically contain multiple part-files rather
    than a single CSV.
    """
    csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    if not csv_files:
        logger.warning(f"No CSV files found in folder: {folder_path}")
    return csv_files


def _assign_category_label(df: pd.DataFrame, folder_name: str) -> pd.DataFrame:
    """
    Attach a 'Category' column (one of config.CLASS_NAMES) to df.

    Behavior:
    - If the raw CSV already has a column named config.RAW_LABEL_COLUMN
      (commonly "label"), that per-row value is assumed to be the specific
      attack name (e.g. "DDoS-UDP_Flood") and is mapped through
      config.ATTACK_FOLDER_TO_CATEGORY. Rows whose raw label is not found in
      the mapping fall back to the folder-based category (see below) with a
      warning, rather than being silently dropped.
    - If no such column exists, every row in this CSV is assigned the
      category derived directly from its parent folder name, since in the
      folder-per-attack release the folder name IS the label.
    """
    folder_category = config.ATTACK_FOLDER_TO_CATEGORY[folder_name]

    if config.RAW_LABEL_COLUMN in df.columns:
        raw_labels = df[config.RAW_LABEL_COLUMN].astype(str)

        def _map_raw_label(raw_label: str) -> str:
            if raw_label in config.ATTACK_FOLDER_TO_CATEGORY:
                return config.ATTACK_FOLDER_TO_CATEGORY[raw_label]
            # Unknown raw label value -> fall back to folder-derived category
            return folder_category

        df["Category"] = raw_labels.map(_map_raw_label)
        # Drop the original raw label column; "Category" is now the single
        # source of truth for the target variable used downstream.
        df = df.drop(columns=[config.RAW_LABEL_COLUMN])
    else:
        df["Category"] = folder_category

    return df


def load_folder(folder_path: str, nrows: int = None) -> pd.DataFrame:
    """
    Load and concatenate every CSV part-file inside a single attack folder
    into one DataFrame, with the 'Category' label column attached.

    Parameters
    ----------
    folder_path : str
        Full path to one attack folder (e.g. .../CICIoT2023/XSS).
    nrows : int, optional
        If provided, only read this many rows PER CSV FILE. Useful for quick
        smoke-testing the pipeline on a laptop before running on the full
        ~13GB dataset.

    Returns
    -------
    pd.DataFrame
        All rows from every CSV in this folder, with feature columns plus
        a 'Category' column.
    """
    folder_name = os.path.basename(os.path.normpath(folder_path))
    csv_files = _find_csv_files(folder_path)

    if not csv_files:
        return pd.DataFrame(columns=config.FEATURE_COLUMNS + ["Category"])

    frames = []
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path, nrows=nrows)
        except Exception as exc:
            logger.error(f"Failed to read {csv_path}: {exc}")
            continue

        df = _assign_category_label(df, folder_name)
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=config.FEATURE_COLUMNS + ["Category"])

    folder_df = pd.concat(frames, ignore_index=True)
    logger.info(
        f"Loaded {len(folder_df):,} rows from {len(csv_files)} file(s) "
        f"in folder '{folder_name}' -> Category = "
        f"'{config.ATTACK_FOLDER_TO_CATEGORY[folder_name]}'"
    )
    return folder_df


def load_all_data(
    raw_dir: str = config.RAW_DATA_DIR,
    nrows_per_file: int = None,
    save_merged_csv: bool = False,
) -> pd.DataFrame:
    """
    Walk every attack folder inside raw_dir, load all CSVs, and return one
    combined DataFrame covering the entire dataset (or a sample of it, if
    nrows_per_file is set).

    Parameters
    ----------
    raw_dir : str
        Root folder containing one sub-folder per attack. Defaults to
        config.RAW_DATA_DIR.
    nrows_per_file : int, optional
        If set, only reads this many rows from EACH individual CSV part-file
        (not per folder). Useful for fast local testing since the full
        dataset is ~13GB / ~46 million rows.
    save_merged_csv : bool, default False
        If True, writes the combined DataFrame to
        data/processed/merged_raw.csv for later re-use, avoiding having to
        re-read all raw CSVs on subsequent runs.

    Returns
    -------
    pd.DataFrame
        Combined dataset with the 46 feature columns plus a 'Category'
        column containing one of config.CLASS_NAMES.
    """
    folders = list_attack_folders(raw_dir)
    logger.info(f"Found {len(folders)} valid attack folders to load.")

    all_frames = []
    for folder_path in folders:
        folder_df = load_folder(folder_path, nrows=nrows_per_file)
        if not folder_df.empty:
            all_frames.append(folder_df)

    if not all_frames:
        raise RuntimeError("No data was loaded from any folder. Aborting.")

    combined_df = pd.concat(all_frames, ignore_index=True)

    logger.info(f"TOTAL rows loaded: {len(combined_df):,}")
    logger.info(
        "Class distribution:\n"
        f"{combined_df['Category'].value_counts().to_string()}"
    )

    if save_merged_csv:
        out_path = os.path.join(config.PROCESSED_DATA_DIR, "merged_raw.csv")
        combined_df.to_csv(out_path, index=False)
        logger.info(f"Saved merged raw dataset to: {out_path}")

    return combined_df


# ---------------------------------------------------------------------------
# Manual smoke test: run `python -m src.data.loader` from the project root
# to quickly verify the loader works against a small sample before running
# it on the full dataset.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = load_all_data(nrows_per_file=500, save_merged_csv=False)
    print(df.head())
    print(df.shape)
    print(df["Category"].value_counts())
