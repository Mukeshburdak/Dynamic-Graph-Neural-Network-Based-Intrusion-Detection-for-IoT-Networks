# IoT Intrusion Detection — Dynamic Graph Neural Networks

A network intrusion detection system for IoT traffic, built on the **CICIoT2023** dataset. Individual network flows are modeled as nodes in a **k-NN similarity graph**, and three Graph Neural Network architectures (**GCN**, **GraphSAGE**, **GAT**) are trained to classify each flow into one of 7 attack categories or Benign — compared directly against classical ML baselines (Random Forest, XGBoost, Logistic Regression) trained on the same features.

## Why a similarity graph, not a topology graph?

CICIoT2023's public CSV release does **not** include Source/Destination IP addresses (removed for privacy), so graphs cannot be built from real network topology. Instead, each flow record becomes a graph node, and edges connect each node to its **k most similar flows** in feature space (Euclidean distance → similarity weight). A sequence of such graphs, built from successive windows of traffic, represents the "dynamic" aspect of the pipeline — new graph snapshots are built as new traffic arrives.

## Project structure

```
iot-intrusion-detection/
├── data/
│   ├── raw/CICIoT2023/        # Downloaded dataset (one folder per attack type)
│   ├── processed/             # Cleaned/balanced/scaled train, val, test CSVs
│   └── graphs/                # Serialized k-NN similarity graph snapshots (.pt)
├── notebooks/                 # Exploratory notebooks (01–05)
├── src/
│   ├── config.py              # Central paths, label mapping, hyperparameters
│   ├── data/                  # loader, preprocessing, feature_selection
│   ├── graph/                 # graph_builder (k-NN graphs), dynamic_graph (windowing)
│   ├── models/                # gcn, graphsage, gat, baseline_ml
│   ├── training/               # train (with early stopping), evaluate
│   └── utils/                 # metrics, visualization
├── models/                    # Saved checkpoints (.pt / .joblib)
├── results/                   # Metrics, figures, comparison.csv
├── dashboard/                 # Flask app: results viewer + live prediction demo
├── tests/                     # pytest unit tests (26 tests, all passing)
├── main.py                    # CLI entry point for the full pipeline
└── requirements.txt
```

## Dataset

Download **CICIoT2023** from the [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/iotdataset-2023.html) and extract it so the folder structure matches:

```
data/raw/CICIoT2023/
├── DDoS-UDP_Flood/*.csv
├── DoS-TCP_Flood/*.csv
├── Mirai-greeth_flood/*.csv
├── Recon-PortScan/*.csv
├── MITM-ArpSpoofing/*.csv
├── XSS/*.csv
├── DictionaryBruteForce/*.csv
├── Benign_Final/*.csv
... (34 attack folders total)
```

The 34 individual attack folders are mapped to **7 categories** (+ Benign) in `src/config.py::ATTACK_FOLDER_TO_CATEGORY`:

| Category | Example folders |
|---|---|
| DDoS | DDoS-UDP_Flood, DDoS-SYN_Flood, DDoS-SlowLoris, ... |
| DoS | DoS-UDP_Flood, DoS-TCP_Flood, DoS-SYN_Flood, DoS-HTTP_Flood |
| Mirai | Mirai-greeth_flood, Mirai-greip_flood, Mirai-udpplain |
| Recon | Recon-PortScan, Recon-OSScan, VulnerabilityScan, ... |
| Spoofing | MITM-ArpSpoofing, DNS_Spoofing |
| Web-based | XSS, SqlInjection, CommandInjection, BrowserHijacking, ... |
| Brute Force | DictionaryBruteForce |
| Benign | Benign_Final |

## Setup

```bash
git clone <repo-url>
cd iot-intrusion-detection
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> **PyTorch / PyTorch Geometric**: the pinned versions in `requirements.txt` assume CPU or standard CUDA 12.1. For GPU acceleration matching your CUDA version, install `torch` from the [official selector](https://pytorch.org/get-started/locally/) first, then the matching `torch-geometric` build, before running `pip install -r requirements.txt`.

## Usage

Everything runs through `main.py`. Start with a fast smoke test on a small data sample before committing to the full ~13GB dataset:

```bash
# Full pipeline on a small sample (minutes, not hours) — sanity-checks everything
python main.py --stage all --nrows 2000

# Full pipeline on the complete dataset
python main.py --stage all
```

Or run each stage independently:

```bash
python main.py --stage preprocess --nrows 5000   # clean, balance, split, scale
python main.py --stage features                  # variance + correlation filtering
python main.py --stage graphs                     # build k-NN similarity graph snapshots
python main.py --stage baseline                   # train Random Forest / XGBoost / LogReg
python main.py --stage train --model gcn --epochs 50
python main.py --stage train --model all           # train GCN + GraphSAGE + GAT
python main.py --stage evaluate                    # test-set evaluation + comparison.csv
```

### Dashboard

```bash
python dashboard/app.py
# open http://127.0.0.1:5000
```

Shows the model comparison table/charts, per-model training curves, confusion matrices, and a live single-flow prediction demo (using the baseline ML models, since a single isolated flow has no graph neighbors for a GNN to use).

### Tests

```bash
pytest tests/ -v
```

26 unit tests covering preprocessing, graph construction, and all three GNN architectures, using small synthetic data (no dataset download required to run tests).

## Pipeline overview

1. **`loader.py`** — walks `data/raw/CICIoT2023/`, reads every CSV per attack folder, labels rows via the folder→category mapping.
2. **`preprocessing.py`** — cleans (drops NaN/inf/duplicates), encodes labels, balances classes (undersample majority / oversample minority — CICIoT2023 has imbalance ratios up to ~5,751:1), stratified 70/15/15 split, scales features (`StandardScaler` fit on train only).
3. **`feature_selection.py`** — drops near-constant and highly-correlated features from the 46 raw features.
4. **`graph_builder.py` / `dynamic_graph.py`** — slides a window over the data, builds a k-NN similarity graph per window (each flow = node, edges = k nearest neighbors by feature distance), saves snapshots to `data/graphs/`.
5. **`gcn.py` / `graphsage.py` / `gat.py`** — three interchangeable GNN architectures sharing the same `forward(x, edge_index, edge_weight)` interface for node-level classification.
6. **`baseline_ml.py`** — Random Forest, XGBoost, Logistic Regression trained on identical features (no graph) as a comparison point.
7. **`train.py`** — shared training loop with class-weighted loss, early stopping on validation macro-F1, checkpointing.
8. **`evaluate.py`** — test-set evaluation for all trained models, merges GNN + baseline results into `results/comparison.csv`.

## Results

After running the full pipeline, results are available in:
- `results/comparison.csv` — accuracy / macro-P/R/F1 for every model
- `results/metrics/` — per-model classification reports and confusion matrices
- `results/figures/` — class distribution, confusion matrices, ROC curves, training curves, graph visualization

## Known limitations / design notes

- **No real network topology**: graphs are built from feature similarity, not actual IP-level connections (CICIoT2023's public CSVs don't include IP addresses).
- **Windowing breaks strict temporal order** post-balancing (class balancing shuffles rows). Graphs represent similarity-based snapshots of traffic batches rather than a strict chronological stream.
- **Dashboard predictions use baseline models**, not GNNs, since single-flow inference has no graph context to aggregate over.

## Acknowledgments

This project was developed by Mukesh Burdak with the assistance of **Claude** (Anthropic) for code architecture, implementation, debugging, and documentation.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.