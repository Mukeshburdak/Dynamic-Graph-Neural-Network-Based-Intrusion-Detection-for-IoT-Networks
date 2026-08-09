"""
main.py

Single command-line entry point for the entire IoT Intrusion Detection
pipeline. Each pipeline stage can be run independently, or the whole
pipeline can be run end-to-end with `--stage all`.

Usage
-----
    # Run everything, start to finish (uses a small row sample for a fast
    # first test -- omit --nrows to use the FULL dataset):
    python main.py --stage all --nrows 2000

    # Run individual stages:
    python main.py --stage preprocess --nrows 2000
    python main.py --stage features
    python main.py --stage graphs
    python main.py --stage baseline
    python main.py --stage train --model gcn --epochs 50
    python main.py --stage train --model all
    python main.py --stage evaluate
"""

import argparse
import logging
import time

from src import config
from src.data.preprocessing import run_preprocessing
from src.data.feature_selection import run_feature_selection
from src.graph.dynamic_graph import build_and_save_graphs_for_split
from src.models.baseline_ml import run_baseline_comparison
from src.training.train import train_model
from src.training.evaluate import evaluate_all_gnn_models, build_final_comparison

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PIPELINE STAGES
# ---------------------------------------------------------------------------

def stage_preprocess(nrows_per_file: int = None):
    logger.info("### STAGE: Preprocessing raw data ###")
    run_preprocessing(nrows_per_file=nrows_per_file)


def stage_features():
    logger.info("### STAGE: Feature selection ###")
    run_feature_selection()


def stage_graphs():
    logger.info("### STAGE: Building dynamic graph snapshots ###")
    splits = [
        (config.TRAIN_CSV_PATH, "train"),
        (config.VAL_CSV_PATH, "validation"),
        (config.TEST_CSV_PATH, "test"),
    ]
    for csv_path, split_name in splits:
        count = build_and_save_graphs_for_split(csv_path, split_name)
        logger.info(f"'{split_name}': {count} graph snapshot(s) created.")


def stage_baseline():
    logger.info("### STAGE: Training classical ML baselines ###")
    run_baseline_comparison()


def stage_train(model_name: str = "all", epochs: int = config.NUM_EPOCHS):
    logger.info(f"### STAGE: Training GNN model(s) [{model_name}] ###")
    models_to_train = ["gcn", "graphsage", "gat"] if model_name == "all" else [model_name]
    for name in models_to_train:
        train_model(name, num_epochs=epochs)


def stage_evaluate():
    logger.info("### STAGE: Evaluating trained models ###")
    gnn_results = evaluate_all_gnn_models()
    if gnn_results.empty:
        logger.warning("No trained GNN checkpoints found -- nothing to evaluate.")
        return
    build_final_comparison(gnn_results)


def run_full_pipeline(nrows_per_file: int = None, model_name: str = "all", epochs: int = config.NUM_EPOCHS):
    """Run every stage of the pipeline in order, start to finish."""
    start_time = time.time()

    stage_preprocess(nrows_per_file)
    stage_features()
    stage_graphs()
    stage_baseline()
    stage_train(model_name, epochs)
    stage_evaluate()

    elapsed_minutes = (time.time() - start_time) / 60
    logger.info(f"### FULL PIPELINE COMPLETE in {elapsed_minutes:.1f} minutes ###")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="IoT Intrusion Detection -- Dynamic GNN pipeline CLI"
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["preprocess", "features", "graphs", "baseline", "train", "evaluate", "all"],
        help="Which pipeline stage to run.",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="If set, only reads this many rows PER raw CSV file during "
             "preprocessing -- useful for a fast smoke test before running "
             "on the full ~13GB dataset. Omit to use all data.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["gcn", "graphsage", "gat", "all"],
        help="Which GNN model to train/evaluate (only used for --stage train).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.NUM_EPOCHS,
        help="Number of training epochs (only used for --stage train/all).",
    )
    args = parser.parse_args()

    if args.stage == "preprocess":
        stage_preprocess(args.nrows)
    elif args.stage == "features":
        stage_features()
    elif args.stage == "graphs":
        stage_graphs()
    elif args.stage == "baseline":
        stage_baseline()
    elif args.stage == "train":
        stage_train(args.model, args.epochs)
    elif args.stage == "evaluate":
        stage_evaluate()
    elif args.stage == "all":
        run_full_pipeline(args.nrows, args.model, args.epochs)


if __name__ == "__main__":
    main()
